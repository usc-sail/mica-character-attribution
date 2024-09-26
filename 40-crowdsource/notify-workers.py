import os
import boto3
import yaml
import pandas as pd

from absl import app
from absl import flags

FLAGS = flags.FLAGS
flags.DEFINE_string("data_dir", default=None, help="data directory", required=True)
flags.DEFINE_string("aws_key_file", default=None, required=True,
					help="file containing AWS access key and secret access key")
flags.DEFINE_string("message", default=None, required=True,
					help="text file containing subject (first line) and message (from second line)")
flags.DEFINE_string("workers", default="40-crowdsource/mturk-workers-qualification/qualification.csv",
					help="csv file of worker ids and their qualification codes")
flags.DEFINE_multi_integer("code", default=[2],
                          help=("qualification codes of the workers to whom the message will be sent. if not given,"
                                " message is passed to all workers"))
flags.DEFINE_multi_string("workerid", default=[],
                          help=("ids of the workers to whom the message will be sent. if this value(s) is passed, then"
                                " the workers and the code arguments are ignored"))
flags.DEFINE_boolean("w", default=False, help="set to do a dry run")

def notify_workers(_):
	aws_key_file = FLAGS.aws_key_file
	message_file = os.path.join(FLAGS.data_dir, FLAGS.message)
	workers_file = os.path.join(FLAGS.data_dir, FLAGS.workers)
	codes = FLAGS.code
	workerids = FLAGS.workerid
	dryrun = FLAGS.w

	with open(aws_key_file) as fr:
		aws_keys = yaml.load(fr, Loader=yaml.FullLoader)

	client = boto3.client("mturk",
						  endpoint_url="https://mturk-requester.us-east-1.amazonaws.com",
						  aws_access_key_id=aws_keys["AWS_ACCESS_KEY_ID"],
						  aws_secret_access_key=aws_keys["AWS_SECRET_ACCESS_KEY"])

	with open(message_file) as fr:
		lines = fr.read().split("\n")
	subject = lines[0]
	message = "\n".join(lines[1:])

	if not workerids:
		workers_df = pd.read_csv(workers_file, index_col=None)
		if codes:
			workerids = (workers_df.loc[workers_df["UPDATE-Character Attribution"].isin(codes), "WorkerId"]
						 .unique().tolist())
		else:
			workerids = workers_df["WorkerId"].unique().tolist()

	print(f"subject ({len(subject)} chars):\n{subject}\n")
	print(f"message ({len(message)} chars):\n{message}\n")
	print(f"{len(workerids)} workers:\n{workerids}\n")
	if not dryrun:
		print("sending emails...")
		response = client.notify_workers(Subject=subject,
							  			 MessageText=message,
							  			 WorkerIds=workerids)
		print("\nresponse:")
		print(response)
	else:
		print("(dry run)")

if __name__ == '__main__':
	app.run(notify_workers)