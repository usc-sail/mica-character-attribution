"""Prompting utilities"""
import datadirs

from absl import flags
from accelerate import PartialState
from accelerate.utils import gather_object, InitProcessGroupKwargs
from datetime import timedelta
from google.api_core import retry
from google import generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import math
import openai
import os
import re
import sys
import tenacity
import tiktoken
import torch
import tqdm
from transformers import BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from vertexai.preview import tokenization
import yaml

# GEMINI
flags.DEFINE_string("gemini_model", default=None, help="Gemini model")
flags.DEFINE_string("gemini_key", default=None, help="Gemini key file")
flags.DEFINE_bool("gemini_count_tokens", default=False, help="Gemini count tokens")

# GPT
flags.DEFINE_string("gpt_model", default=None, help="GPT model")
flags.DEFINE_string("gpt_key", default=None, help="OpenAI key file")
flags.DEFINE_bool("gpt_count_tokens", default=False, help="GPT count tokens")

# HUGGINGFACE
flags.DEFINE_string("hf_model",
                    default=None,
                    help="Huggingface model or local path to model directory relative to datadir")
flags.DEFINE_bool("bf16", default=False, help="use brain float-16 precision (default=float-16)")
flags.DEFINE_bool("load_4bit", default=False, help="load in 4 bit")
flags.DEFINE_bool("load_8bit", default=False, help="load in 8 bit")
flags.DEFINE_enum("attn",
                  default="sdpa",
                  enum_values=["flash_attention_2", "sdpa", "eager"],
                  help="attention implementation")

# GENERATION STRATEGIES
flags.DEFINE_bool("replace_system_role",
                  default=False,
                  help="substitute system role with user role if system role is not supported by tokenizer")
flags.DEFINE_integer("batch_size", default=1, help="batch size")
flags.DEFINE_integer("max_input_tokens", default=None, help="maximum tokens per input sequence in K (=2^10)")
flags.DEFINE_integer("max_output_tokens", default=1, help="maximum tokens to generate")
flags.DEFINE_bool("do_sample", default=False, help="use sampling; otherwise use greedy decoding")
flags.DEFINE_integer("top_k",
                     default=None,
                     help="number of highest probability vocabulary tokens to keep for top-k filtering")
flags.DEFINE_float("top_p",
                   default=None,
                   help=("If set to float < 1, only the smallest set of most probable tokens with probabilities that "
                         "add up to top_p or higher are kept for generation"))
flags.DEFINE_float("temperature", default=1, help="temperature for generations")

def models_checker(args):
    gpt = args["gpt_model"] is not None
    gemini = args["gemini_model"] is not None
    hf = args["hf_model"] is not None
    return not (gpt and gemini and hf)

def gpt_key_checker(args):
    gpt = args["gpt_model"] is not None
    key = args["gpt_key"] is not None
    return not gpt or key

def gemini_key_checker(args):
    gemini = args["gemini_model"] is not None
    key = args["gemini_key"] is not None
    return not gemini or key

flags.register_multi_flags_validator(["gpt_model", "gemini_model", "hf_model"],
                                     models_checker,
                                     message="Provide exactly one of gemini or hf models")
flags.register_multi_flags_validator(["gemini_model", "gemini_key"],
                                     gemini_key_checker,
                                     message="Provide API key file if you are using gemini model")
flags.register_multi_flags_validator(["gpt_model", "gpt_key"],
                                     gpt_key_checker,
                                     message="Provide API key file if you are using gpt model")

FLAGS = flags.FLAGS

def modelname():
    if FLAGS.gpt_model is not None:
        return FLAGS.gpt_model
    if FLAGS.gemini_model is not None:
        return FLAGS.gemini_model
    if FLAGS.hf_model is not None:
        model_name_or_path = os.path.join(datadirs.datadir, FLAGS.hf_model)
        if os.path.exists(model_name_or_path):
            model_name_or_path = re.sub(os.path.join(datadirs.datadir, "50-modeling/finetune/") + "/",
                                        "",
                                        model_name_or_path)
            model_name_or_path = model_name_or_path.replace("/", "--")
        else:
            model_name_or_path = re.sub(r"^[^/]+/", "", FLAGS.hf_model)
        bf16 = "-bf16" if FLAGS.bf16 else ""
        quant = "-4bit" if FLAGS.load_4bit else "-8bit" if FLAGS.load_8bit else ""
        return f"{model_name_or_path}{bf16}{quant}"
    raise ValueError("model not provided")

class GPT:
    """GPT generator"""
    def __init__(self):
        print("configuring OpenAI API")
        gpt_key_file = FLAGS.gpt_key
        with open(gpt_key_file) as fr:
            gpt_key_dict = yaml.load(fr, Loader=yaml.FullLoader)
        self.client = openai.Client(api_key=gpt_key_dict["key"],
                                    organization=gpt_key_dict["organization"],
                                    project=gpt_key_dict["project"])
        self.encoding = tiktoken.encoding_for_model(FLAGS.gpt_model)

    @tenacity.retry(wait=tenacity.wait_random_exponential(min=1, max=60), stop=tenacity.stop_after_attempt(10))
    def completion_with_backoff(self, **kwargs):
        return self.client.chat.completions.create(**kwargs)

    def __call__(self, prompts, system_instr=None):
        if FLAGS.gpt_count_tokens:
            ntokens_arr = [len(self.encoding.encode(prompt)) for prompt in tqdm.tqdm(prompts, desc="counting tokens")]
            print(f"{sum(ntokens_arr)} input tokens, {FLAGS.max_output_tokens * len(prompts)} output tokens")
            user_input = input("Do you want to proceed with prompting? [y/n] ")
            if user_input.lower().strip() != "y":
                sys.exit(0)
        responses = []
        for prompt in tqdm.tqdm(prompts, desc="prompting"):
            messages = [{"role": "developer", "content": system_instr}] if system_instr is not None else []
            messages.append({"role": "user", "content": prompt})
            output = self.completion_with_backoff(messages=messages,
                                                  model=FLAGS.gpt_model,
                                                  max_completion_tokens=FLAGS.max_output_tokens,
                                                  temperature=FLAGS.temperature,
                                                  top_p=FLAGS.top_p)
            try:
                response = output.choices[0].message.content
            except Exception:
                response = "ERROR"
            responses.append(response)
        return responses

class Gemini:
    """Gemini generator"""
    def __init__(self, system_instr=None):
        print("configuring Gemini API")
        gemini_key_file = FLAGS.gemini_key
        with open(gemini_key_file) as fr:
            gemini_api_key = fr.read().strip()
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name=FLAGS.gemini_model,
                                           system_instruction=system_instr)
        self.tokenizer = tokenization.get_tokenizer_for_model(FLAGS.gemini_model)

    def __call__(self, prompts):
        if FLAGS.gemini_count_tokens:
            ntokens_arr = [self.tokenizer.count_tokens(prompt).total_tokens
                           for prompt in tqdm.tqdm(prompts, desc="counting tokens")]
            print(f"{sum(ntokens_arr)} input tokens, {FLAGS.max_output_tokens * len(prompts)} output tokens")
            user_input = input("Do you want to proceed with prompting? [y/n] ")
            if user_input.lower().strip() != "y":
                sys.exit(0)
            
        safety_settings={HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE}
        generation_config = genai.GenerationConfig(temperature=FLAGS.temperature,
                                                   max_output_tokens=FLAGS.max_output_tokens,
                                                   top_k=FLAGS.top_k,
                                                   top_p=FLAGS.top_p)
        request_opts = genai.types.RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300))
        responses = []
        for prompt in tqdm.tqdm(prompts, desc="prompting"):
            output = self.model.generate_content(prompt,
                                                 safety_settings=safety_settings,
                                                 generation_config=generation_config,
                                                 request_options=request_opts)
            try:
                response = output.candidates[0].content.parts[0].text
            except Exception:
                response = "ERROR"
            responses.append(response)
        return responses

class HF:
    """Huggingface model generator"""
    def __init__(self):
        kwargs = InitProcessGroupKwargs(timeout=timedelta(hours=12)).to_kwargs()
        self.partial_state = PartialState(**kwargs)
        self.partial_state.print("instantiating model")
        compute_dtype = torch.bfloat16 if FLAGS.bf16 else torch.float16
        if FLAGS.load_4bit:
            quantization_config = BitsAndBytesConfig(load_in_4bit=True,
                                                     bnb_4bit_compute_dtype=compute_dtype,
                                                     bnb_4bit_quant_storage=compute_dtype,
                                                     bnb_4bit_quant_type="nf4",
                                                     bnb_4bit_use_double_quant=True)
        elif FLAGS.load_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
        else:
            quantization_config = None
        model_name_or_path = os.path.join(datadirs.datadir, FLAGS.hf_model)
        if not os.path.exists(model_name_or_path):
            model_name_or_path = FLAGS.hf_model
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                          torch_dtype=compute_dtype,
                                                          quantization_config=quantization_config,
                                                          device_map={"": self.partial_state.process_index},
                                                          attn_implementation=FLAGS.attn,
                                                          trust_remote_code=True)
        self.partial_state.print(f"generation config: {self.model.generation_config}")
        self.partial_state.print("instantiating tokenizer")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       padding_side="left",
                                                       truncation_side="right",
                                                       trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

    def __call__(self, prompts, system_instr=None):
        if system_instr is not None:
            if FLAGS.replace_system_role:
                messages = [[{"role": "user", "content": f"{system_instr}\n{prompt}"}] for prompt in prompts]
            else:
                messages = [[{"role": "system", "content": system_instr}, {"role": "user", "content": prompt}]
                            for prompt in prompts]
            prompts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        self.partial_state.wait_for_everyone()
        with self.partial_state.split_between_processes(prompts, apply_padding=True) as process_prompts:
            n_batches = math.ceil(len(process_prompts)/FLAGS.batch_size)
            process_responses = []
            for i in range(n_batches):
                batch_prompts = process_prompts[i * FLAGS.batch_size: (i + 1) * FLAGS.batch_size]
                batch_encoding = (self
                                  .tokenizer(batch_prompts,
                                             padding="max_length" if FLAGS.max_input_tokens is not None else "longest",
                                             truncation=FLAGS.max_input_tokens is not None,
                                             return_tensors="pt",
                                             max_length=((1 << 10) * FLAGS.max_input_tokens 
                                                         if FLAGS.max_input_tokens is not None else None))
                                  .to(self.partial_state.device))
                batch_output = self.model.generate(**batch_encoding,
                                                   do_sample=FLAGS.do_sample,
                                                   top_k=FLAGS.top_k,
                                                   top_p=FLAGS.top_p,
                                                   temperature=FLAGS.temperature,
                                                   max_new_tokens=FLAGS.max_output_tokens,
                                                   pad_token_id=self.tokenizer.pad_token_id,
                                                   return_legacy_cache=True)
                batch_output = batch_output[:, batch_encoding["input_ids"].shape[1]:]
                batch_responses = self.tokenizer.batch_decode(batch_output, skip_special_tokens=True)
                process_responses.extend(batch_responses)
                print(f"PROCESS {self.partial_state.process_index}: Batch {i + 1} / {n_batches} done")
            responses = [process_responses]
        responses_arr = gather_object(responses)
        responses = [response for responses in responses_arr for response in responses]
        responses = responses[:len(prompts)]
        return responses