from jinja2 import Template
from natsort import natsorted
from pathlib import Path
import os

pwd = (Path(os.getcwd()) / Path(__file__)).parent
hf_checkpoint_prefix = Path("/home/ane53vq/storage/modernbert/hf_conversion")

jobs = []

for m in [
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/120M_LLM2Vec_2048/iter-01430512-ckpt_120M_mntp_sim_new",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/120M_LLM2Vec_2048/iter-01430512-ckpt_120M_sim_new",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLM2Vec_2048/iter-01430512-ckpt_1B_mntp_sim_new",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLM2Vec_2048/iter-01430512-ckpt_1B_sim_new",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/7B_LLM2Vec_2048/iter-01430512-ckpt_7B_mntp_sim",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/7B_LLM2Vec_2048/iter-01430512-ckpt_7B_sim",
    ]:
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:2}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:4}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:8}"])

# alte Modelle
for m in [
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/120M_LLM2Vec_mntp-10000_simcse-200",  
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLM2Vec_mntp-10000_simcse-200",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/7B_llammlein_fa3/models-hf/iter-01430512-ckpt_mntp-10000_simcse-200",
    ]:
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:4}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:8}"])
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}", "+model.model_config_args.rope_scaling={type:'dynamic',factor:16}"])

for m in [
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLaMA_LLM2Vec/Llama-3.2-1B_llama1B_mntp_sim",
    "/data/42-julia-hpc-rz-lsx/juw57zv/models/1B_LLaMA_LLM2Vec/Llama-3.2-1B_llama1B_sim"
    ]: 
    jobs.append(["train_args=a100", "+task=niah_germanquad", "+model=llm2vec_for_mntp-10000_simcse-200", f"model.model_name={m}"])

with open('slurm_template.jinja') as f:
    tmpl = Template(f.read())
print(tmpl.render(pwd=pwd, jobs=jobs))


