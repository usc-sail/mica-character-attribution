"""Prompting utilities"""
import datadirs

from absl import flags
from absl import logging
from google.api_core import retry
from google import generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import os
import re
import sys
import torch
from torch.utils.data import Dataset
import tqdm
from transformers import pipeline, BitsAndBytesConfig, AutoTokenizer, AutoModelForCausalLM
from vertexai.preview import tokenization

flags.DEFINE_string("gemini_model", default=None, help="Gemini model")
flags.DEFINE_string("gemini_key", default=None, help="Gemini key file")
flags.DEFINE_bool("gemini_estimate_cost", default=False, help="Gemini estimate cost")
flags.DEFINE_string("llama_model", default=None, help="Llama model or path to model directory")
flags.DEFINE_integer("batch_size", default=1, help="batch size")
flags.DEFINE_integer("max_input_tokens", default=128, help="maximum tokens per input sequence in K (=2^10)")
flags.DEFINE_integer("max_output_tokens", default=256, help="maximum tokens to generate")
flags.DEFINE_string("padding", default="longest", help="padding strategy")
flags.DEFINE_float("temperature", default=1, help="temperature for generations")
flags.DEFINE_bool("bf16", default=False, help="use brain float-16 precision (default=float-16)")
flags.DEFINE_bool("load_4bit", default=False, help="load in 4 bit")
flags.DEFINE_bool("load_8bit", default=False, help="load in 8 bit")
flags.DEFINE_bool("flash_attn", default=False, help="use flash-attention")
flags.DEFINE_string("device", default="auto", help="cuda device to use")

def models_checker(args):
    gemini = args["gemini_model"] is not None
    llama = args["llama_model"] is not None
    return not (gemini and llama)

def key_checker(args):
    gemini = args["gemini_model"] is not None
    key = args["gemini_key"] is not None
    return not gemini or key

flags.register_multi_flags_validator(["gemini_model", "llama_model"], models_checker,
                                     message="Provide exactly one of gemini or llama models")
flags.register_multi_flags_validator(["gemini_model", "gemini_key"], key_checker,
                                     message="Provide API key file if you are using gemini model")

FLAGS = flags.FLAGS

def modelname():
    if FLAGS.llama_model is not None:
        model_name_or_path = os.path.join(datadirs.datadir, FLAGS.llama_model)
        if os.path.exists(model_name_or_path):
            model_name_or_path = re.sub(os.path.join(datadirs.datadir, "50-modeling/finetune/"), "", model_name_or_path)
            model_name_or_path = model_name_or_path.replace("/", "--")
        else:
            model_name_or_path = re.sub("meta-llama/", FLAGS.llama_model)
        bf16 = "-bf16" if FLAGS.bf16 else ""
        quant = "-4bit" if FLAGS.load_4bit else "-8bit" if FLAGS.load_8bit else ""
        return f"{model_name_or_path}{bf16}{quant}"
    if FLAGS.gemini_model is not None:
        return FLAGS.gemini_model
    raise ValueError("model not provided")

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

class Gemini:
    """Gemini generator"""
    def __init__(self, system_instr=None):
        logging.info("configuring Gemini API")
        gemini_key_file = FLAGS.gemini_key
        with open(gemini_key_file) as fr:
            gemini_api_key = fr.read().strip()
        genai.configure(api_key=gemini_api_key)
        self.model = genai.GenerativeModel(model_name=FLAGS.gemini_model,
                                           system_instruction=system_instr)
        self.tokenizer = tokenization.get_tokenizer_for_model(FLAGS.gemini_model)
        self.pro = "pro" in FLAGS.gemini_model

    def __call__(self, prompts):
        if FLAGS.gemini_estimate_cost:
            if self.pro:
                input_token_rate, output_token_rate = 1.25, 5
            else:
                input_token_rate, output_token_rate = 0.075, 0.3
            ntokens_arr = [self.tokenizer.count_tokens(prompt).total_tokens
                           for prompt in tqdm.tqdm(prompts, desc="counting tokens")]
            cost = (sum(ntokens_arr) * input_token_rate
                    + len(prompts) * FLAGS.max_output_tokens * output_token_rate)/(1<<17)
            logging.info(f"estimated cost = ${cost:.2f}")
            user_input = input("Do you want to proceed with prompting? [y/n] ")
            if user_input.lower().strip() != "y":
                sys.exit(0)
            
        safety_settings={HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                         HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                         HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                         HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE}
        generation_config = genai.GenerationConfig(temperature=FLAGS.temperature,
                                                   max_output_tokens=FLAGS.max_output_tokens)
        request_opts = genai.types.RequestOptions(retry=retry.Retry(initial=10, multiplier=2, maximum=60, timeout=300))
        for prompt in tqdm.tqdm(prompts, desc="prompting"):
            output = self.model.generate_content(prompt,
                                                 safety_settings=safety_settings,
                                                 generation_config=generation_config,
                                                 request_options=request_opts)
            try:
                response = output.candidates[0].content.parts[0].text
            except Exception:
                response = "ERROR"
            yield response

class Llama:
    """Llama generator"""
    def __init__(self):
        logging.info("instantiating model")
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
        model_name_or_path = os.path.join(datadirs.datadir, FLAGS.llama_model)
        if not os.path.exists(model_name_or_path):
            model_name_or_path = FLAGS.llama_model
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                     torch_dtype=compute_dtype,
                                                     quantization_config=quantization_config,
                                                     device_map=FLAGS.device,
                                                     attn_implementation=("flash_attention_2" if FLAGS.flash_attn else
                                                                          "sdpa"))
        logging.info("instantiating tokenizer")
        length = (1 << 10) * FLAGS.max_input_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path,
                                                       padding_side="left",
                                                       truncation_side="right",
                                                       model_max_length=length)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        logging.info("instantiating pipeline")
        self.generator = pipeline(task="text-generation", model=model, tokenizer=self.tokenizer)

    def __call__(self, prompts, system_instr=None, **kwargs):
        if system_instr is not None:
            messages = [[{"role": "system", "content": system_instr}, {"role": "user", "content": prompt}]
                        for prompt in prompts]
            prompts = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompt_dataset = TextDataset(prompts)
        for output in tqdm.tqdm(self.generator(prompt_dataset,
                                               batch_size=FLAGS.batch_size,
                                               temperature=FLAGS.temperature,
                                               return_full_text=False,
                                               max_new_tokens=FLAGS.max_output_tokens,
                                               padding=FLAGS.padding,
                                               truncation=True,
                                               **kwargs),
                                desc="prompting",
                                total=len(prompt_dataset),
                                miniters=1):
            response = output[0]["generated_text"]
            sys.stdout.flush()
            yield response        