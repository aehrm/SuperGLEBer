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

ext1_models = list(natsorted(hf_checkpoint_prefix.glob("modernbert_1b_ext1_helma_43218*")))[-1:]
ext2_models = list(natsorted(hf_checkpoint_prefix.glob("modernbert_1b_ext2_helma_44604*")))[-1:]
pretrain_model = "modernbert_1b_middle_helma_313949--ep0-ba158000-rank0"

jobs = []
for m in ext1_models:
    jobs.append(make_job(hf_checkpoint_prefix / m))

for m in ext2_models:
    jobs.append(make_job(hf_checkpoint_prefix / m))

jobs.append(make_job(hf_checkpoint_prefix / pretrain_model))
jobs.append(make_job(hf_checkpoint_prefix / pretrain_model, "+model.model_config_args.global_rope_theta=160e3"))

for m in [
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/120M_LLM2Vec_2048/iter-01430512-ckpt_120M_mntp_sim_new",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/120M_LLM2Vec_2048/iter-01430512-ckpt_120M_sim_new",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLM2Vec_2048/iter-01430512-ckpt_1B_mntp_sim_new",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLM2Vec_2048/iter-01430512-ckpt_1B_sim_new",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/7B_LLM2Vec_2048/iter-01430512-ckpt_7B_mntp_sim",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/7B_LLM2Vec_2048/iter-01430512-ckpt_7B_sim",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLaMA_LLM2Vec/Llama-3.2-1B_llama1B_mntp_sim",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLaMA_LLM2Vec/Llama-3.2-1B_llama1B_sim"
    ]:
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:2}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:4}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:8}"])


with open('slurm_template.jinja') as f:
    tmpl = Template(f.read())
print(tmpl.render(pwd=pwd, jobs=jobs))


