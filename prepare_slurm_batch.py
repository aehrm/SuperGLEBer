from jinja2 import Template
from natsort import natsorted
from pathlib import Path
import os

pwd = (Path(os.getcwd()) / Path(__file__)).parent
hf_checkpoint_prefix = Path("/home/ane53vq/storage/modernbert/hf_conversion")

def make_job(model_path, *args):
    job = ["train_args=a100", "+task=niah_germanquad", "+model=modern_gbert", f"model.model_name={str(model_path)}"]
    job.extend(args)
    return job

ext1_models = list(natsorted(hf_checkpoint_prefix.glob("modernbert_1b_ext1_helma_43218*")))[-1::-3]
ext2_models = list(natsorted(hf_checkpoint_prefix.glob("modernbert_1b_ext2_helma_44604*")))[-1::-2]
pretrain_model = "modernbert_1b_middle_helma_313949--ep0-ba158000-rank0"

jobs = []
for m in ext1_models:
    jobs.append(make_job(hf_checkpoint_prefix / m))

for m in ext2_models:
    jobs.append(make_job(hf_checkpoint_prefix / m))

jobs.append(make_job(hf_checkpoint_prefix / pretrain_model))
jobs.append(make_job(hf_checkpoint_prefix / pretrain_model, "+model.model_config_args.global_rope_theta=160e3"))

with open('slurm_template.jinja') as f:
    tmpl = Template(f.read())
print(tmpl.render(pwd=pwd, jobs=jobs))


