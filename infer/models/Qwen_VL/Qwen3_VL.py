from transformers import Qwen3VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch


class Qwen3_VL:
    def __init__(self, model_path):
        super().__init__()
        self.llm = Qwen3VLForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map={"": 0},                 
            attn_implementation="sdpa",
            tp_plan=None,                      
        )
        self.processor = AutoProcessor.from_pretrained(model_path)

    def process_messages(self, messages):
        prompt = messages["prompt"]
        start_frame_path = messages["start_frame_path"]
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": 'image_url', "image_url": start_frame_path
                    },
                    {
                        "type": "text", "text": prompt
                    }
                ]
            }
        ]
        prompt = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[prompt],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")

        return inputs

    def generate_output(self, messages):
        inputs = self.process_messages(messages)
        generated_ids = self.llm.generate(**inputs, do_sample=False, repetition_penalty=1, max_new_tokens=8192)
        generated_ids_trimmed = [
            out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        
        return output_text[0]
    
    def generate_outputs(self, messages_list):
        return [self.generate_output(messages) for messages in messages_list]

''' template
class YourOwnModel
    def __init__(self, model_path):
        # load your model and processor here
    def generate_outputs(self, messages_list):
        # input: a list of messages
        # output the model answers in a list
'''