import json
import openai
import tiktoken
import json
import os
import re
import time
from openai import AzureOpenAI
from virtualhome.simulation.unity_simulator.comm_unity import UnityCommunication

enc = tiktoken.get_encoding("cl100k_base")
with open('../../secrets.json') as f:
    credentials = json.load(f)

dir_system = './system'
dir_prompt = './prompt'
dir_query = './query'
prompt_load_order = ['prompt_role',
                     'prompt_function',
                     'prompt_environment',
                     'prompt_output_format',
                     'prompt_example']


def reset(comm, scene_index=None):
    response = comm.post_command({'id': str(time.time()), 'action': 'reset', 'intParams': [
    ] if scene_index is None else [scene_index]})
    return response['success']


def generate_script(input_array):
    output_array = []
    for action in input_array:

        action = action.replace(">", "").replace("<", "").replace(" ", "")
        # Split the action into its constituent parts
        parts = action.split('(')
        verb = parts[0].lower()
        arguments = parts[1].strip(')')
        # Check if there are any objects
        if len(arguments) == 0:
            objects = []
        else:
            objects = arguments.split(',')
            objects = [obj.split('_') for obj in objects]
        # Create the output string
        if len(objects) == 0:
            output_array.append('<char0> [{}]'.format(verb))
        elif len(objects) == 1:
            obj_type, obj_id = objects[0]
            output_array.append(
                '<char0> [{}] <{}> ({})'.format(
                    verb, obj_type, obj_id))
        else:
            obj1_type, obj1_id = objects[0]
            obj2_type, obj2_id = objects[1]
            output_array.append(
                '<char0> [{}] <{}> ({}) <{}> ({})'.format(
                    verb, obj1_type, obj1_id, obj2_type, obj2_id))

    return output_array


def remove_brackets(name):
    return name.replace('<', '').replace('>', '')


def which_room(graph, node_id):
    # Create a mapping from each node ID to its corresponding node data
    id_to_node = {node['id']: node for node in graph['nodes']}
    # Create a mapping from child node ID to its parent node ID
    child_to_parent = {}
    for edge in graph['edges']:
        if edge['from_id'] in child_to_parent.keys():
            child_to_parent[edge['from_id']].append(
                (edge['to_id'], edge['relation_type']))
        else:
            child_to_parent[edge['from_id']] = [
                (edge['to_id'], edge['relation_type'])]
    if node_id not in child_to_parent.keys():
        return None
    # Find the parent node ID(s) of the input node
    parent_node_ids = child_to_parent[node_id]
    # Iterate over all parent node IDs
    for parent_node_id in parent_node_ids:
        # Check if the parent node is a room
        if 'Rooms' in id_to_node[parent_node_id[0]]['category']:
            # Return the name of the room
            return id_to_node[parent_node_id[0]]['class_name']
    # If no room is found, return None
    return None


def find_parent_node(graph, node_name, room_name):
    # Create a mapping from each node ID to its corresponding node data
    id_to_node = {node['id']: node for node in graph['nodes']}
    name_to_id = {}
    for node in graph['nodes']:
        if node['class_name'] in name_to_id.keys():
            name_to_id[node['class_name']].append(node['id'])
        else:
            name_to_id[node['class_name']] = [node['id']]
    child_to_parent = {}
    for edge in graph['edges']:
        if edge['from_id'] in child_to_parent.keys():
            child_to_parent[edge['from_id']].append(
                (edge['to_id'], edge['relation_type']))
        else:
            child_to_parent[edge['from_id']] = [
                (edge['to_id'], edge['relation_type'])]
    if '_' in node_name:
        node_ids = [int(node_name.split('_')[1])]
        node_name = node_name.split('_')[0]
    else:
        # Find the node ID of the input node name
        if node_name not in name_to_id.keys():
            return None
        node_ids = name_to_id[node_name]
        # print(node_ids)
        node_ids = [
            node_id for node_id in node_ids if which_room(
                graph, node_id) == room_name]
        # print(node_ids)
    return_dict = {"object_states": {}, "asset_states": {}}
    for node_id in node_ids:
        if 'GRABBABLE' in id_to_node[node_id]['properties']:
            key_to_add = "object_states"
        else:
            key_to_add = "asset_states"
        # Find the parent node ID(s) of the input node
        if node_id not in child_to_parent.keys():
            return None
        else:
            parent_node_ids = child_to_parent[node_id]

        # Iterate over all parent node IDs
        for parent_node_id in parent_node_ids:
            parent_node = id_to_node[parent_node_id[0]]
            relation_type = parent_node_id[1]
            # focus only in and on
            if relation_type != 'INSIDE' and relation_type != 'ON':
                continue
            if 'Decor' in parent_node['category']:
                continue
            #print(parent_node['class_name'], parent_node_id[1])
            if "<{}_{}>".format(node_name,
                                node_id) in return_dict[key_to_add].keys():
                return_dict[key_to_add]["<{}_{}>".format(node_name, node_id)].append(
                    "{}(<{}_{}>)".format(relation_type, parent_node['class_name'], parent_node_id[0]))
            else:
                return_dict[key_to_add]["<{}_{}>".format(node_name, node_id)] = ["{}(<{}_{}>)".format(
                    relation_type, parent_node['class_name'], parent_node_id[0])]
    return return_dict


def populate_environment(graph, start_objects, start_room):
    environment = {
        "assets": [],
        "asset_states": {},
        "objects": [],
        "object_states": {},
    }
    # Create a mapping from each node ID to its corresponding node data
    id_to_node = {node['id']: node for node in graph['nodes']}
    # note that there are multiple nodes with the same name
    name_to_id = {}
    for node in graph['nodes']:
        if node['class_name'] in name_to_id.keys():
            name_to_id[node['class_name']].append(node['id'])
        else:
            name_to_id[node['class_name']] = [node['id']]
    # Create a mapping from child node ID to its parent node ID
    child_to_parent = {}
    for edge in graph['edges']:
        if edge['from_id'] in child_to_parent.keys():
            child_to_parent[edge['from_id']].append(
                (edge['to_id'], edge['relation_type']))
        else:
            child_to_parent[edge['from_id']] = [
                (edge['to_id'], edge['relation_type'])]
    objects_to_check = [remove_brackets(name) for name in start_objects]

    while objects_to_check:
        current_object = objects_to_check.pop()
        # print(objects_to_check)
        if "<{}>".format(current_object) not in environment["objects"] and "<{}>".format(
                current_object) not in environment["assets"]:
            # add to the environment
            if 'GRABBABLE' in id_to_node[int(
                    current_object.split('_')[-1])]['properties']:
                environment["objects"].append("<{}>".format(current_object))
            else:
                environment["assets"].append("<{}>".format(current_object))

            # find the parent and add the relationship to the environment
            parent_info = find_parent_node(
                graph, remove_brackets(current_object), start_room)
            if parent_info is not None:
                if "object_states" in parent_info:
                    for obj, states in parent_info["object_states"].items():
                        # add states to the environment
                        environment["object_states"]["<{}>".format(remove_brackets(obj))] = ["{}(<{}>)".format(
                            state.split('(')[0], remove_brackets(state.split('(')[-1].split(')')[0])) for state in states]
                        # add the new objects involved in the states to the
                        # list of objects to check
                        for state in states:
                            involved_object = remove_brackets(
                                state.split('(')[-1].split(')')[0])
                            if "<{}>".format(involved_object) not in environment["objects"] and "<{}>".format(
                                    involved_object) not in environment["assets"]:
                                objects_to_check.append(involved_object)
                if "asset_states" in parent_info:
                    for obj, states in parent_info["asset_states"].items():
                        # add states to the environment
                        environment["asset_states"]["<{}>".format(remove_brackets(obj))] = ["{}(<{}>)".format(
                            state.split('(')[0], remove_brackets(state.split('(')[-1].split(')')[0])) for state in states]
                        # add the new assets involved in the states to the list
                        # of assets to check
                        for state in states:
                            # remove brackets while keeping the ID
                            involved_asset = remove_brackets(state.split(
                                '(')[-1].split(')')[0])  # remove the ID and brackets
                            if "<{}>".format(involved_asset) not in environment["assets"] and "<{}>".format(
                                    involved_asset) not in environment["objects"]:
                                objects_to_check.append(involved_asset)
    # want to add 'object_properties' to the environment
    asset_properties = {}
    for asset in environment['asset_states']:
        asset_id = asset.strip('>').strip('<').split('_')[1]
        tmp_properties = []
        if "CAN_OPEN" in id_to_node[int(asset_id)]['properties']:
            tmp_properties.append("IS_OPENABLE")
        else:
            tmp_properties.append("NOT_OPENABLE")
        asset_properties[asset] = tmp_properties
    environment['asset_properties'] = asset_properties
    object_properties = {}
    for obj in environment['object_states']:
        obj_id = obj.strip('>').strip('<').split('_')[1]
        tmp_properties = []
        if "CAN_OPEN" in id_to_node[int(obj_id)]['properties']:
            tmp_properties.append("IS_OPENABLE")
        else:
            tmp_properties.append("NOT_OPENABLE")
        object_properties[obj] = tmp_properties
    environment['object_properties'] = object_properties
    return environment


def find_unique_objects(graph, object_name, start_room):
    hit_object = find_parent_node(graph, object_name, start_room)
    if hit_object is None:
        return []
    if len(hit_object['object_states']) > 0:
        object_list = hit_object['object_states'].keys()
    elif len(hit_object['asset_states']) > 0:
        object_list = hit_object['asset_states'].keys()
    else:
        # error
        raise ValueError('No object found')
    return list(object_list)


def extract_objects(script):
    objects_all = []
    for action in script:
        parts = action.split('(')
        arguments = parts[1].replace(" ", "").strip(')')
        # Check if there are any objects
        if len(arguments) == 0:
            objects = []
        else:
            objects = arguments.split(',')
        objects_all.extend(objects)
    return list(set(objects_all))


class ChatGPT:
    VALID_API_VERSIONS = ['2022-12-01', '2023-05-15']

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
        # json part is between ```python and ```.
        # skip if there is no json part
        if text.find('```python') == -1:
            return text
        text_json = text[text.find(
            '```python') + len('```python'):text.find('\n```')]
        text_json.replace('```', '')
        return text_json

    def generate(self, message, environment, is_user_feedback=False):
        if is_user_feedback:
            self.messages.append({'sender': 'user',
                                  'text': message})
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
        # print(json.dumps(speech_result, indent=4))
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
        # print(response.to_json())
        text = response.to_dict()['choices'][0]["message"]["content"]
        print("-> content " + "*" * 100)

        self.last_response_raw = text
        self.messages.append(
            {"sender": "assistant", "text": self.last_response_raw})
        # analyze the response
        self.last_response = text
        self.last_response = self.extract_json_part(self.last_response)
        self.last_response = self.last_response.replace("'", "\"")
        try:
            self.json_dict = json.loads(self.last_response, strict=False)
            self.json_dict["environment_before"] = environment
            self.environment = self.json_dict["environment_after"]
        except BaseException as e:
            import pdb
            pdb.set_trace()
            self.json_dict = None
            return None
        return self.json_dict


def test_execution(comm, script):
    reset(comm)
    print('Starting scene...')
    comm.add_character('Chars/Male2', initial_room='kitchen')
    # Running the script step by step
    for i, script_atom in enumerate(script):
        task = text["task_cohesion"]["task_sequence"][i]
        task_atom = script_atom.split("[")[1].split("]")[0]
        ret = comm.render_script([script_atom], frame_rate=10, recording=True)
        # If the simulation fails, return an error message.
        if ret[0] is False:
            if 'putin' in script_atom:
                # This is for cases where the simulation fails due to a 'putin' action on an openable object.
                # The 'putin' action is only allowed if the object has been opened beforehand.
                # Therefore, if the simulation fails, it's likely that the user
                # didn't open the object before attempting to put something
                # into it.
                if ('microwave' in script_atom) or (
                        'stove' in script_atom) or ('fridge' in script_atom):
                    if i > 0 and ('open' not in script[i - 1]):
                        return "You are wrong! Modify your answer. You need to open and close an openable object when you 'putin' something into it."
            # In other cases, return an error message.
            feedback = "You are wrong! Modify your answer. The following line failed in a simulator: " + task + "\n" + \
                "The verb {" + task_atom + "} is not applicable to the object(s). Refer to \'HUMAN ACTION LIST\' in my instruction."
            return feedback
    # The following code is for cases where the simulation succeeds but the sequence of actions is incorrect.
    # Incorrect case1: skipping the 'open' action before 'putin' action for
    # openable objects.
    for i, script_atom in enumerate(script):
        if ('put' in script_atom) or ('putin' in script_atom):
            if ('microwave' in script_atom) or (
                    'stove' in script_atom) or ('fridge' in script_atom):
                if i > 0 and ('open' not in script[i - 1]):
                    return "You are wrong! Modify your answer. You need to open and close an openable object when you 'putin' something into it."
    # Incorrect case2: 'switchon' action before 'putin' action in warmup task.
    for i, script_atom in enumerate(script):
        if ('put' in script_atom) or ('putin' in script_atom):
            if ('microwave' in script_atom) or ('stove' in script_atom):
                index_switchon = [
                    j for j, s in enumerate(script) if (
                        'switchon' in s and (
                            'microwave' in s or 'stove' in s))]
                if len(index_switchon) > 0 and index_switchon[0] < i:
                    return "You are wrong! Modify your answer. Before you turn on the switch, please put the object you want to warm inside."
    # Incorrect case3: forget to turn on the switch in warmup task.
    for i, script_atom in enumerate(script):
        if ('put' in script_atom) or ('putin' in script_atom):
            if ('microwave' in script_atom) or ('stove' in script_atom):
                if not (any('switchon' in s.lower() for s in script)):
                    return "You are wrong! Modify your answer. Do not forget to turn on the switch in the end."
    return ""


if __name__ == '__main__':
    comm = UnityCommunication()
    dir_name = "out_feedback_test_gpt-4o_temp=0.7"
    waittime_sec = 5  # wait 5 seconds between api calls to mitigate the rate limit
    max_trial = 6
    time_api_called = time.time() - waittime_sec
    for scenario_id in range(1, 15):  # start from 1 until 14
        trial_idx = 0
        user_feedback = ""
        while trial_idx < max_trial:
            print(f"scenario_id={scenario_id}, trial_idx={trial_idx}")
            scenario_name = 'scenario_' + str(scenario_id)
            dump_name = './' + dir_name + f'/{scenario_name}/{trial_idx}'
            fp = os.path.join(dump_name + '.json')
            if os.path.exists(fp):
                # if the file exists, skip this scenario
                print(f"skip scenario_id={scenario_id}")
                break

            with open('scenarios/' + str(scenario_id) + '.json') as f:
                scenario = json.load(f)
            instructions = scenario['instructions']
            reference_program = scenario['program']
            print(
                f"instructions(scenario_id={scenario_id}): {instructions[0]}")
            reset(comm)
            s, graph = comm.environment_graph()
            environment = populate_environment(
                graph, extract_objects(reference_program), "kitchen")
            scenario_name = 'scenario_' + str(scenario_id)
            if not os.path.exists('./' + dir_name + '/' + scenario_name):
                os.makedirs('./' + dir_name + '/' + scenario_name)
            while True:
                # if api is called within waittime_sec, wait
                current_time = time.time()
                if current_time - time_api_called < waittime_sec:
                    print("waiting for " + str(waittime_sec - \
                          (current_time - time_api_called)) + " seconds...")
                    time.sleep(waittime_sec - (current_time - time_api_called))
                if trial_idx == 0:
                    aimodel = ChatGPT(prompt_load_order=prompt_load_order, use_azure=True)
                    text = aimodel.generate(
                        instructions[0],
                        environment,
                        is_user_feedback=False)
                else:  # trial_idx > 0: # use user feedback
                    assert user_feedback != ""
                    text = aimodel.generate(
                        user_feedback, environment, is_user_feedback=True)
                    time_api_called = time.time()
                if text is not None:
                    break
                else:
                    print("api call failed. retrying...")
                    current_time = time.time()
                    if current_time - time_api_called < waittime_sec:
                        print("waiting for " + str(waittime_sec - \
                              (current_time - time_api_called)) + " seconds...")
                        time.sleep(waittime_sec -
                                   (current_time - time_api_called))
                    text = aimodel.generate(
                        "Your return cannot be interpreted as a valid json dictionary. Please reformat your response.",
                        environment,
                        is_user_feedback=True)
                    break
            if text is None:
                trial_idx = 5
                dump_name = './' + dir_name + f'/{scenario_name}/note'
                fp = os.path.join(dump_name + '.txt')
                # In the file, note that the trial was skipped
                with open(fp, 'w') as f:
                    f.write(aimodel.last_response)
                break
            print("self test is running...")
            script = generate_script(text["task_cohesion"]["task_sequence"])
            user_feedback = test_execution(comm, script)
            if len(user_feedback) > 0:
                # VirtualHome sometimes fails to execute the script even if the
                # script is correct, so retry once just in case.
                user_feedback = test_execution(comm, script)
            print('result of self test: ' + user_feedback)
            was_execution_successful = False
            if len(user_feedback) > 0:
                was_execution_successful = False
            else:
                was_execution_successful = True

            dump_name = './' + dir_name + f'/{scenario_name}/{trial_idx}'
            fp = os.path.join(dump_name + '.json')
            aimodel.json_dict['was_execution_successful'] = was_execution_successful
            aimodel.json_dict['user_feedback'] = user_feedback
            with open(fp, 'w') as f:
                json.dump(aimodel.json_dict, f, indent=4)
            if not was_execution_successful:  # if execution was not successful, retry using the feedback
                trial_idx += 1
                if trial_idx == max_trial:
                    break
            else:
                trial_idx = 5
                break
