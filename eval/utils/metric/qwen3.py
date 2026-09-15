import transformers
import torch
from transformers import AutoTokenizer
from utils.metric.utils import *

class Qwen:
    def __init__(self, model, device="mps"):

        self.tokenizer = AutoTokenizer.from_pretrained(
            model,
        )

        if self.tokenizer.eos_token_id is None and self.tokenizer.eos_token is not None:
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)

        if self.tokenizer.eos_token_id is None:
            if self.tokenizer.eos_token is None:
                self.tokenizer.eos_token = "</s>"
            self.tokenizer.eos_token_id = self.tokenizer.convert_tokens_to_ids(self.tokenizer.eos_token)

        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
            tokenizer=self.tokenizer,  
        )

        if hasattr(self.pipeline, "model"):
            model_obj = self.pipeline.model
            if hasattr(model_obj, "config"):
                model_obj.config.eos_token_id = self.tokenizer.eos_token_id
                if hasattr(model_obj, "generation_config") and model_obj.generation_config is not None:
                    model_obj.generation_config.eos_token_id = self.tokenizer.eos_token_id

    def evaluate(self, gts, preds, relations, relation_preds):
        scores = []
        relation_scores = []
        for phrase, sentence in zip(gts, preds):
            prompt = prompt_template.format(sentence=sentence, phrase=phrase)
            if len(sentence) == 0:
                scores.append(0)
                continue
            score = self.generate(prompt)
            scores.append(score)

        for relation, relation_sentence in zip(relations, relation_preds):
            if len(relation_sentence) == 0:
                relation_scores.append(0)
                continue
            prompt = relation_prompt_template.format(sentence=relation_sentence, phrase=relation)
            score = self.generate(prompt)
            relation_scores.append(score)

        return scores, relation_scores
    
    def generate(self, prompt):
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": prompt},
        ]

        chat_str = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False, # Switches between thinking and non-thinking modes. Default is True.
        )

        try:
            eot_id = self.tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if eot_id == self.tokenizer.unk_token_id:
                eot_id = None
        except Exception:
            eot_id = None

        terminators = [self.tokenizer.eos_token_id]
        if eot_id is not None and eot_id != self.tokenizer.eos_token_id:
            terminators.append(eot_id)

        # Tokenize first and then truncate to ensure the input does not exceed the model’s maximum context length.
        model_max = getattr(self.tokenizer, "model_max_length", 131072)
        if model_max is None or model_max > 10**9:
            model_max = 131072

        max_new = 256
        max_input = max(1, model_max - max_new)

        enc = self.tokenizer(
            chat_str,
            return_tensors="pt",
            truncation=True,
            max_length=max_input,
        )

        device = self.pipeline.model.device
        enc = {k: v.to(device) for k, v in enc.items()}

        generated_ids = self.pipeline.model.generate(
            **enc,
            do_sample=False,
            max_new_tokens=max_new,
            eos_token_id=terminators,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        input_len = enc["input_ids"].shape[-1]
        gen_text = self.tokenizer.decode(
            generated_ids[0][input_len:],
            skip_special_tokens=True
        )

        score = get_first_digit(gen_text)
        return score
