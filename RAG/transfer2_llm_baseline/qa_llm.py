import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class QA_LLM:
    def __init__(self):
        self.B_INST, self.E_INST = "[INST]", "[/INST]"
        self.B_SYS, self.E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

    def load_model(self,llm_model_name = 'elyza/ELYZA-japanese-Llama-2-7b-fast-instruct',quantization_config=None):
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoModelForCausalLM.from_pretrained(llm_model_name, torch_dtype="auto",quantization_config=quantization_config, device_map="auto")

    def load_model_LoRA(self,llm_model_name = 'elyza/ELYZA-japanese-Llama-2-7b-fast-instruct', model_lora_name = 'output_LoRA', quantization_config=None):
        from peft import AutoPeftModelForCausalLM
        self.tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.model = AutoPeftModelForCausalLM.from_pretrained(model_lora_name, torch_dtype="auto",quantization_config=quantization_config, device_map="auto")
        print(self.model)

    def set_default_sys_prompt(self,prompt):
        self.DEFAULT_SYSTEM_PROMPT = prompt
        

    def generate_output(self,text,max_new_tokens=256,after_inst=''):
        prompt = "{bos_token}{b_inst} {system}{prompt} {e_inst}{after_inst}".format(
            bos_token=self.tokenizer.bos_token,
            b_inst=self.B_INST,
            system=f"{self.B_SYS}{self.DEFAULT_SYSTEM_PROMPT}{self.E_SYS}",
            prompt=text,
            e_inst=self.E_INST,
            after_inst=after_inst,
        )

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        output_ids = self.model.generate(
            token_ids.to(self.model.device),
            max_new_tokens=max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
        return output

