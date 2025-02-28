import openai
import tiktoken
import json
import os
import re
import argparse
import sys
import rclpy
# append meta_action path
sys.path.append("../..")
from meta_action import meta_actions
from meta_action.meta_actions import PlanExecutor
from openai import AzureOpenAI

enc = tiktoken.get_encoding("cl100k_base")

dir_system = './system'
dir_prompt = './prompt'
dir_query = './query'
prompt_load_order = ['prompt_role',
                     'prompt_function',
                     'prompt_output_format',
                     'prompt_example']


# azure openai api has changed from '2023-05-15' to '2023-05-15'
# if you are using a 0301 version, use '2022-12-01'
# Otherwise, use '2023-05-15'

class ChatGPT:

    def __init__(
            self,
            prompt_load_order,
            use_azure=True,
            api_version='2024-05-01-preview'):
        self.use_azure = use_azure
        if self.use_azure:
            self.endpoint = os.getenv("ENDPOINT_URL", "https://perception-openai.openai.azure.com/")  
            self.deployment = os.getenv("DEPLOYMENT_NAME", "gpt-4o")
            self.subscription_key = os.getenv("AZURE_OPENAI_API_KEY", "")  

            # 使用基于密钥的身份验证来初始化 Azure OpenAI 客户端
            self.client = AzureOpenAI(  
                azure_endpoint=self.endpoint,  
                api_key=self.subscription_key,  
                api_version="2024-05-01-preview",  
            )

        self.messages = []
        self.max_token_length = 16000
        self.max_completion_length = 1000
        self.last_response = None
        self.query = ''
        self.instruction = ''
        self.user_feedback = 'Please adjust your output based on the feedback.'
        # load prompt file
        fp_system = os.path.join(dir_system, 'system.txt')
        with open(fp_system) as f:
            data = f.read()
        self.system_message = {"role": "system", "content": data}

        # load prompt file
        for prompt_name in prompt_load_order:
            fp_prompt = os.path.join(dir_prompt, prompt_name + '.txt')
            with open(fp_prompt) as f:
                data = f.read()
            data_spilit = re.split(r'\[user\]\n|\[assistant\]\n', data)
            data_spilit = [item for item in data_spilit if len(item) != 0]
            # it start with user and ends with system
            assert len(data_spilit) % 2 == 0
            for i, item in enumerate(data_spilit):
                if i % 2 == 0:
                    self.messages.append({"sender": "user", "text": item})
                else:
                    self.messages.append({"sender": "assistant", "text": item})
        fp_query = os.path.join(dir_query, 'query.txt')
        with open(fp_query) as f:
            self.query = f.read()

    # See
    # https://learn.microsoft.com/en-us/azure/cognitive-services/openai/how-to/chatgpt#chatml
    def create_prompt(self):
        prompt = []
        prompt.append(self.system_message)
        for message in self.messages:
            prompt.append(
                {"role": message['sender'], "content": message['text']})
        prompt_content = ""
        for message in prompt:
            prompt_content += message["content"]
        print('prompt length: ' + str(len(enc.encode(prompt_content))))
        if len(enc.encode(prompt_content)) > self.max_token_length - \
                self.max_completion_length:
            print('prompt too long. truncated.')
            # truncate the prompt by removing the oldest two messages
            self.messages = self.messages[2:]
            prompt = self.create_prompt()
        return prompt

    def extract_json_part(self, text):
        "because the json part is in the middle of the text, we need to extract it. " \
        "json part is between ``` and ```."

        # skip if there is no json part
        if text.find('```') == -1:
            return text
        # text_json = text[text.find(
        #     '```') + 3:text.find('```', text.find('```') + 3)]
        text_json = text[text.find("{"): text.find('```', text.find('```') + 3)]
        return text_json

    def generate(self, message, is_user_feedback=False):
        if is_user_feedback:
            self.messages.append({'sender': 'user',
                                  'text': message + "\n" + self.user_feedback})
        else:
            text_base = self.query
            # if text_base.find('[ENVIRONMENT]') != -1:
            #     text_base = text_base.replace(
            #         '[ENVIRONMENT]', json.dumps(environment))
            if text_base.find('[INSTRUCTION]') != -1:
                text_base = text_base.replace('[INSTRUCTION]', message)
                self.instruction = text_base
            self.messages.append({'sender': 'user', 'text': text_base})
        
        speech_result = self.create_prompt()
        print("--> prompt " + '=' * 100)
        print(json.dumps(speech_result, indent=4))
        response = self.client.chat.completions.create(  
            model=self.deployment,  
            messages=speech_result,  
            max_tokens=800,  
            temperature=0.7,  
            top_p=0.95,  
            frequency_penalty=0,  
            presence_penalty=0,  
            stop=None,  
            stream=False  
        )  
        # print("-> response" + '-' * 100)
        # print(response.to_json())
        text = response.to_dict()['choices'][0]["message"]["content"]
        # print("-> content " + "*" * 100)
        # print(text)
        self.last_response = text
        self.last_response = self.extract_json_part(self.last_response)
        # self.last_response = self.last_response.replace("'", "\"")
        print("-> extracted content " + "=" * 100)
        print(self.last_response)
        # dump to a text file
        with open('last_response.txt', 'w') as f:
            f.write(self.last_response)
        try:
            self.json_dict = json.loads(self.last_response, strict=False)
            # self.environment = self.json_dict["environment_after"]
        except BaseException as e:
            import pdb
            pdb.set_trace()
            self.json_dict = None

        if len(self.messages) > 0 and self.last_response is not None:
            self.messages.append(
                {"sender": "assistant", "text": self.last_response})

        return self.json_dict

    def dump_json(self, dump_name=None):
        if dump_name is not None:
            # dump the dictionary to json file dump 1, 2, ...
            fp = os.path.join(dump_name + '.json')
            with open(fp, 'w') as f:
                json.dump(self.json_dict, f, indent=4)


def plan_exec(result):
    task_sequence = result['task_cohesion']['task_sequence']
    # object_name = result['task_cohesion']['object_name'].strip("<>")
    # print(task_sequence, object_name)

    rclpy.init()
    # task_sequence = [
    #     "search_roughly_and_approach_object()",
    #     "move_chassis_based_on_object()",
    #     "detect_precisely()",
    #     "grasp_object()"
    # ]
    object_name = "shoe"
    plan_executor = PlanExecutor(task_sequence, object_name)
    exec_result = plan_executor.execute()
    # rclpy.spin(plan_executor)
    plan_executor.destroy_node()
    rclpy.shutdown()
    return (exec_result, task_sequence)

def feedback_handle_exception(failed_action, task_sequence):
    "generate user feedback automaticlly while exception occering during 'failed_action' in 'task_sequence'"
    feedback = "The original task_sequence is:\n" + f"{json.dumps( dict(task_sequence=task_sequence))}\n" \
               f"While executing {failed_action} in above task_sequence, an exception occurred. Please give me a new plan dictionary, to complete the original task_sequence." \
               "Adhere to the output format I defined in the above instruction. Follow the six rules. Think step by step."
    return feedback

def main():
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--input', required=True, type=str, help="please input the instruction.")
    # args = parser.parse_args()
    # instruction = args.input

    aimodel = ChatGPT(
        prompt_load_order=prompt_load_order,
        use_azure=True)

    if not os.path.exists(f'./out_{aimodel.deployment.replace("-", "_")}/'):
        os.makedirs(f"./out_{aimodel.deployment.replace('-', '_')}/")

    while True:
        instruction = input("please input instruction (input empty to quit):")
        if instruction == '':
            break
        result = aimodel.generate(
            instruction,
            is_user_feedback=False)
        while True:
            user_feedback = input(
                'user feedback (return empty if satisfied): ')
            if user_feedback == 'q':
                exit()
            if user_feedback == 'r':
                break
            if user_feedback != '':
                result = aimodel.generate(
                    user_feedback, is_user_feedback=True)
            else:
                plan_result = plan_exec(result)
                print(plan_result)
                break

        # while True:
        #     plan_result = plan_exec(result)
        #     if plan_result[0] is None:
        #         aimodel.dump_json(f"./out_{aimodel.deployment.replace('-', '_')}/{instruction.replace(' ', '_')}")
        #         break
        #     else:
        #         user_feedback = feedback_handle_exception(plan_result[0], plan_result[1])
        #         result = aimodel.generate(user_feedback, is_user_feedback=True)


if __name__ == "__main__":
    main()
    # plan_exec(1)
