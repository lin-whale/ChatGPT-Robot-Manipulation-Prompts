import openai
import tiktoken
import json
import os
import re
import argparse
from openai import AzureOpenAI  

enc = tiktoken.get_encoding("cl100k_base")

dir_system = './system'
dir_prompt = './prompt'
dir_query = './query'
prompt_load_order = ['prompt_role',
                     'prompt_function',
                     'prompt_environment',
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
        # because the json part is in the middle of the text, we need to extract it.
        # json part is between ``` and ```.
        # skip if there is no json part
        if text.find('```') == -1:
            return text
        # text_json = text[text.find(
        #     '```') + 3:text.find('```', text.find('```') + 3)]
        text_json = text[text.find("{"): text.find('```', text.find('```') + 3)]
        return text_json

    def generate(self, message, environment, is_user_feedback=False):
        if is_user_feedback:
            self.messages.append({'sender': 'user',
                                  'text': message + "\n" + self.instruction})
        else:
            text_base = self.query
            if text_base.find('[ENVIRONMENT]') != -1:
                text_base = text_base.replace(
                    '[ENVIRONMENT]', json.dumps(environment))
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
        print("-> response" + '-' * 100)
        print(response.to_json())
        text = response.to_dict()['choices'][0]["message"]["content"]
        print("-> content " + "*" * 100)
        print(text)
        self.last_response = text
        self.last_response = self.extract_json_part(self.last_response)
        self.last_response = self.last_response.replace("'", "\"")
        print("-> extracted content " + "=" * 100)
        print(self.last_response)
        # dump to a text file
        with open('last_response.txt', 'w') as f:
            f.write(self.last_response)
        try:
            self.json_dict = json.loads(self.last_response, strict=False)
            self.environment = self.json_dict["environment_after"]
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--scenario',
        type=str,
        required=True,
        help='scenario name (see the code for details)')
    args = parser.parse_args()
    scenario_name = args.scenario
    # 1. example of moving objects on the table and the shelf
    if scenario_name == 'shelf':
        environment = {
            "assets": [
                "<table>",
                "<shelf_bottom>",
                "<shelf_top>",
                "<trash_bin>",
                "<floor>"],
            "asset_states": {
                "<shelf_bottom>": "on_something(<table>)",
                "<trash_bin>": "on_something(<floor>)"},
            "objects": [
                "<spam>",
                "<juice>"],
            "object_states": {
                "<spam>": "on_something(<table>)",
                "<juice>": "on_something(<shelf_bottom>)"}}
        instructions = ['Put the juice on top of the shelf',
                        'Throw away the spam into the trash bin',
                        'Move the juice on top of the table',
                        'Throw away the juice']
    # 2. example of opening and closing the fridge, and putting the juice on
    # the floor
    elif scenario_name == 'fridge':
        environment = {
            "assets": [
                "<fridge>",
                "<floor>"],
            "asset_states": {
                "<fridge>": "on_something(<floor>)"},
            "objects": [
                "<fridge_handle>",
                "<juice>"],
            "object_states": {
                "<fridge_handle>": "closed()",
                "<juice>": "inside_something(<fridge>)"}}
        instructions = ['Open the fridge half way',
                        'Open the fridge wider',
                        'Take the juice in the fridge and put it on the floor',
                        'Close the fridge']
    # 3. example of opening and closing the drawer
    elif scenario_name == 'drawer':
        environment = {"assets": ["<drawer>", "<floor>"],
                       "asset_states": {"<drawer>": "on_something(<floor>)"},
                       "objects": ["<drawer_handle>"],
                       "object_states": {"<drawer_handle>": "closed()"}}
        instructions = ['Open the drawer widely',
                        'Close the drawer half way',
                        'Close the drawer fully']
    # 4. example of wiping the table
    elif scenario_name == 'table':
        environment = {
            "assets": [
                "<table1>",
                "<table2>",
                "<trash_bin>",
                "<floor>"],
            "asset_states": {
                "<table1>": "next_to(<table2>)",
                "<trash_bin>": "on_something(<floor>)"},
            "objects": ["<sponge>"],
            "object_states": {
                "<sponge>": "on_something(<table1>)"}}
        instructions = ['Put the sponge on the table2',
                        'Wipe the table2 with the sponge']
    # 5. example of wiping the window
    elif scenario_name == 'window':
        environment = {
            "assets": [
                "<table>",
                "<window>",
                "<trash_bin>",
                "<floor>"],
            "asset_states": {
                "<table>": "next_to(<window>)",
                "<trash_bin>": "on_something(<floor>)"},
            "objects": ["<sponge>"],
            "object_states": {
                "<sponge>": "on_something(<table>)"}}
        instructions = [
            'Get the sponge from the table and wipe the window with it. After that, put the sponge back on the table',
            'Throw away the sponge on the table']
    else:
        parser.error('Invalid scenario name:' + scenario_name)

    aimodel = ChatGPT(
        prompt_load_order=prompt_load_order,
        use_azure=True)

    if not os.path.exists(f'./out_{aimodel.deployment.replace('-', '_')}/' + scenario_name):
        os.makedirs(f'./out_{aimodel.deployment.replace('-', '_')}/' + scenario_name)
    for i, instruction in enumerate(instructions):
        print(json.dumps(environment))
        text = aimodel.generate(
            instruction,
            environment,
            is_user_feedback=False)
        while True:
            user_feedback = input(
                'user feedback (return empty if satisfied): ')
            # user_feedback = ''
            if user_feedback == 'q':
                exit()
            if user_feedback != '':
                text = aimodel.generate(
                    user_feedback, environment, is_user_feedback=True)
            else:
                # update the current environment
                environment = aimodel.environment
                break
        aimodel.dump_json(f'./out_{aimodel.deployment.replace('-', '_')}/{scenario_name}/{i}')
