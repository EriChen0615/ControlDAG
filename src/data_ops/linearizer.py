from abc import ABC
from abc import abstractmethod
import json
import sys
sys.path.append('third_party/')
from google_nlg.utterance_generator import TemplateUtteranceGenerator
from collections import defaultdict
import os
from tqdm import tqdm

class LinearizerBase(ABC):
    def __init__(self, input_field, output_field):
        self.input_field = input_field
        self.output_field = output_field

    @abstractmethod
    def tokenize_from_acts(self, dialogue_act_list) -> str:
        return ""

    def __call__(self, example):
        return {self.output_field: self.tokenize_from_acts(example[self.input_field])}

class SGD_NaiveLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, **kwargs):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map
    
    def tokenize_from_acts(self, dialogue_act_list) -> str:
        res = ''
        for dialog_act in dialogue_act_list:
            # res += self.act_idx_name_map[dialog_act['act']] + ' ' + dialog_act['slot'] +  ' ' + ' '.join(dialog_act['values']) + ' <SEP> '
            res += f"{self.act_idx_name_map[dialog_act['act']]} {dialog_act['slot']} {' '.join(dialog_act['values'])} <SEP> "
        return res[:-7] # remove trailing <SEP>

class SGD_PaperNaiveLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, **kwargs):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map

    def tokenize_from_acts(self, dialogue_act_list) -> str:
        res = ''
        for dialog_act in dialogue_act_list:
            if dialog_act['values']:
                # res += self.act_idx_name_map[dialog_act['act']] + ' ( ' + dialog_act['slot'] + ' = ' + ' '.join(dialog_act['values']) +  ' ) ' + ' '
                res += f"{self.act_idx_name_map[dialog_act['act']]} ( {dialog_act['slot']} = {' '.join(dialog_act['values'])} ) "
            elif dialog_act['slot']:
                # res += self.act_idx_name_map[dialog_act['act']] + ' ( ' + dialog_act['slot'] + ' ) ' 
                res += f"{self.act_idx_name_map[dialog_act['act']]} ( {dialog_act['slot']} ) "
            else:
                res += f"{self.act_idx_name_map[dialog_act['act']]} "

        return res[:-1] # remove trailing space

class SGD_SepNaiveLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, separator=" && ", **kwargs):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map
        self.separator = separator

    def tokenize_from_acts(self, dialogue_act_list) -> str:
        res = ''
        for dialog_act in dialogue_act_list:
            if dialog_act['values']:
                # res += self.act_idx_name_map[dialog_act['act']] + ' ( ' + dialog_act['slot'] + ' = ' + ' '.join(dialog_act['values']) +  ' ) ' + ' '
                res += f"{self.act_idx_name_map[dialog_act['act']]} ( {dialog_act['slot']} = {self.separator.join(dialog_act['values'])} ) "
            elif dialog_act['slot']:
                # res += self.act_idx_name_map[dialog_act['act']] + ' ( ' + dialog_act['slot'] + ' ) ' 
                res += f"{self.act_idx_name_map[dialog_act['act']]} ( {dialog_act['slot']} ) "
            else:
                res += f"{self.act_idx_name_map[dialog_act['act']]} "

        return res[:-1] # remove trailing space


class SGD_SchemaGuidedLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, schema_paths):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map
        schema_arr = []
        for schema_file in schema_paths:
            with open(schema_file, 'r') as f:
                schema_arr.extend(json.loads(f.read()))

        self.service_desc_dict = {}
        self.schema_slots_desc_dict = {}
        self.service_uncategorical_slots = defaultdict(set)
        self.service_boolean_slots = defaultdict(set)
        for schema in schema_arr:
            service_name = schema['service_name']
            self.service_desc_dict[service_name] = schema['description']
            self.schema_slots_desc_dict[service_name] = {}
            for slot in schema['slots']:
                self.schema_slots_desc_dict[service_name][slot['name']] = slot['description']
                if not slot['is_categorical']:
                    self.service_uncategorical_slots[service_name].add(slot['name'])
                elif "True" in slot['possible_values']:
                    self.service_boolean_slots[service_name].add(slot['name'])

    def tokenize_from_acts(self, dialogue_act_list, service) -> str:
        res = ''
        for dialog_act in dialogue_act_list:
            act_name = self.act_idx_name_map[dialog_act['act']]
            if not service in self.schema_slots_desc_dict:
                print("service not in schema", service, dialog_act['slot'], dialog_act)
                continue
            if not dialog_act['slot'] in self.schema_slots_desc_dict[service] and not dialog_act['slot'] in {'count', 'intent', ''}:
                print("ERROR! slot doesn't exist in schema", service, dialog_act['slot'])
                continue
            slot_desc = self.schema_slots_desc_dict[service][dialog_act['slot']] if not dialog_act['slot'] in {'', 'count', 'intent'} else dialog_act['slot'] # if slot = '' or 'count' or 'intent', keep it as it is
            if dialog_act['values']:
                res += f"{act_name} ( {slot_desc} = {' '.join(dialog_act['values'])} ) "
            elif dialog_act['slot']:
                res += f"{act_name} ( {slot_desc} ) "
            else:
                res += f"{act_name} "
        return res[:-1]
    
    def tokenize_slots(self, dialog_act, service):
        res = ''
        act_name = dialog_act['act'] 
        if not service in self.schema_slots_desc_dict:
            print("service not in schema", service, dialog_act['slot'], dialog_act)
            return ""
        if not dialog_act['slot'] in self.schema_slots_desc_dict[service] and not dialog_act['slot'] in {'count', 'intent', ''}:
            print("ERROR! slot doesn't exist in schema", service, dialog_act['slot'])
            return ""
        slot_desc = self.schema_slots_desc_dict[service][dialog_act['slot']] if not dialog_act['slot'] in {'', 'count', 'intent'} else dialog_act['slot'] # if slot = '' or 'count' or 'intent', keep it as it is
        if dialog_act['values']:
            res += f"{act_name} ( {slot_desc} = {' '.join(dialog_act['values'])} ) "
        elif dialog_act['slot']:
            res += f"{act_name} ( {slot_desc} ) "
        else:
            res += f"{act_name} "
        return res[:-1]
    
    def get_slot_desc(self, dialog_act, service):
        act_name = dialog_act['act'] 
        if not service in self.schema_slots_desc_dict:
            print("service not in schema", service, dialog_act['slot'], dialog_act)
            return ""
        if not dialog_act['slot'] in self.schema_slots_desc_dict[service] and not dialog_act['slot'] in {'count', 'intent', ''}:
            print("ERROR! slot doesn't exist in schema", service, dialog_act['slot'])
            return ""
        slot_desc = self.schema_slots_desc_dict[service][dialog_act['slot']] if not dialog_act['slot'] in {'', 'count', 'intent'} else dialog_act['slot'] # if slot = '' or 'count' or 'intent', keep it as it is
        return slot_desc
    
    def get_hypos_by_slots(self, example):
        res = []
        service = example['service']
        diag_actions = example['dialog_acts']
        for action in diag_actions:
            act_name = self.act_idx_name_map[action['act']]
            if not service in self.schema_slots_desc_dict:
                print("service not in schema", service, action['slot'], action)
                continue
            if act_name == "OFFER_INTENT":
                continue
            elif act_name == "INFORM_COUNT":
                count_num = action['values'][0]
                res.append((action, f"There is 1 thing" if count_num == "1" else f"There are {count_num} things"))
                continue
            elif action['slot'] == '': # empty slot name suggests it is a pleasantry intent
                continue
            elif not action['slot'] in self.schema_slots_desc_dict[service]:
                print("ERROR! slot doesn't exist in schema", service, action)
            else:
                hypo = ""
                slot_desc = self.schema_slots_desc_dict[service][action['slot']]
                if action['slot'] in self.service_boolean_slots[service]: # turn into a question for boolean slots
                    if len(action['values']) == 0:
                        continue
                    else:
                        hypo = f"{slot_desc}? " + ("Yes." if action['values'][0] == "True" else "No.")
                elif act_name == "REQUEST" and len(action["values"]) == 0: # request a slot
                    hypo = f"Request {slot_desc}"
                else:
                    if len(action['values']) == 1:
                        hypo = f"{slot_desc} is {action['values'][0]}"
                    elif act_name == "REQUEST":
                        hypo = f"{slot_desc} are {' or '.join(action['values'])}"
                    else:
                        hypo = f"{slot_desc} are {', '.join(action['values'])}"
                res.append((action, hypo))
        return res
    
    def get_multiple_hypos_by_slots(self, example):
        res = []
        service = example['service']
        domain = service.split('_')[0]
        diag_actions = example['dialog_acts']
        for action in diag_actions:
            act_name = self.act_idx_name_map[action['act']] if isinstance(action['act'], int) else action['act']
            if not service in self.schema_slots_desc_dict:
                print("service not in schema", service, action['slot'], action)
                continue
            if act_name == "OFFER_INTENT":
                continue
            elif act_name == "INFORM_COUNT":
                hypos = []
                count_num = action['values'][0]
                hypos.append(f"There is 1 thing" if count_num == "1" else f"There are {count_num} things")
                hypos.append(f"There is 1 {domain}" if count_num == "1" else f"There are {count_num} {domain}")
                res.append((action, hypos))
                continue
            elif action['slot'] == '' or (act_name=="REQUEST" and action['slot']=='intent'): # empty slot name suggests it is a pleasantry intent OR request intent
                continue
            elif not action['slot'] in self.schema_slots_desc_dict[service]:
                print("ERROR! slot doesn't exist in schema", service, action)
            else:
                hypo = ""
                slot_desc = self.schema_slots_desc_dict[service][action['slot']]
                hypos = []
                if action['slot'] in self.service_boolean_slots[service]: # turn into a question for boolean slots
                    if len(action['values']) == 0:
                        continue
                    else:
                        bool_val = True if action['values'][0] == "True" else False
                        slot_name = action['slot'].split('_')
                        hypos.append(f"{slot_desc}? " + ("Yes." if bool_val else "No.")) # QA Formulation
                        hypos.append(f"{' '.join(slot_name)}? " + ("Yes." if bool_val else "No.")) # QA Formulation
                        # slot formulation
                        if "has" in slot_name or "is" in slot_name or "have" in slot_name: # the slot name itself is sensible
                            if bool_val:
                                hypos.append(' '.join(slot_name))
                                slot_name.insert(0, domain)
                                hypos.append(' '.join(slot_name))
                            elif "has" in slot_name or "have" in slot_name:
                                hypos.append(f"Does not {' '.join(slot_name)}")
                                hypos.append(f"{domain} does not {' '.join(slot_name)}")
                            else:
                                slot_name.insert(1, 'not')
                                hypos.append(' '.join(slot_name))
                                hypos.append(f"{domain} {' '.join(slot_name)}")
                        else: # need prefix verb
                            if bool_val:
                                prefs = ["Has", "Is", "Does"]
                            else:
                                prefs = ["Has no", "Is not", "Does not"]
                            for pr in prefs:
                                hypos.append(f"{pr} {' '.join(slot_name)}")
                        res.append((action, hypos))
                elif act_name == "REQUEST" and len(action["values"]) == 0: # request a slot
                    hypos.append(f"Request {slot_desc}")
                    res.append((action, hypos))
                else:
                    slot_name = ' '.join(action['slot'].split('_'))
                    if len(action['values']) == 1:
                        hypos.append(f"{slot_desc} is {action['values'][0]}")
                        hypos.append(f"{slot_name} is {action['values'][0]}")
                    elif act_name == "REQUEST":
                        hypos.append(f"{slot_desc} are {' or '.join(action['values'])}")
                        hypos.append(f"{slot_name} are {' or '.join(action['values'])}")
                    else:
                        hypos.append(f"{slot_desc} are {', '.join(action['values'])}")
                        hypos.append(f"{slot_name} are {', '.join(action['values'])}")
                    res.append((action, hypos))
        return res

    # override __call__ as we need additional information
    def __call__(self, example):
        return {self.output_field: self.tokenize_from_acts(example[self.input_field], example['service'])}

class SGD_SchemaGuidedWithServiceLinearizer(SGD_SchemaGuidedLinearizer):
    def __init__(self, act_idx_name_map, schema_paths):
        super().__init__(act_idx_name_map, schema_paths)
    
    def tokenize_from_acts(self, dialog_act_list, service) -> str:
        res = super().tokenize_from_acts(dialog_act_list, service)
        res = f"SERVICE ( {service} = {self.service_desc_dict[service]} ) " + res
        return res

class SGD_TemplateGuidedLinearizer(LinearizerBase):
    def __init__(self, act_idx_name_map, dataset_dir, template_dir):
        super().__init__("dialog_acts", '_linearized')
        self.act_idx_name_map = act_idx_name_map
        self.utter_generator = TemplateUtteranceGenerator(template_dir)
        # construct mapping to additional information. Key of the form {split}-{dialogue_id}-{turn_id}
        self.dialogue_id_turn_id_info_map = defaultdict(map)
        json_filename = f"data/dialogue_id_turn_id_info_map-{template_dir.split('/')[-1]}.json" 
        if os.path.exists(json_filename):
            with open(json_filename, 'r') as f:
                print("Read extra info map for t2g2 lineaizer from memory")
                self.dialogue_id_turn_id_info_map = json.loads(f.read())
        else:
            for split in ['train', 'test', 'dev']:
                data_dir = f"{dataset_dir}/{split}"
                for filename in tqdm(os.listdir(data_dir)):
                    if filename.startswith('dialogue') and filename.endswith('json'):
                        dialogues = None
                        with open(os.path.join(data_dir, filename), 'r') as f:
                            dialogues = json.loads(f.read())
                        for diag in dialogues:
                            diag_id = diag['dialogue_id']
                            for turn_id, turn in enumerate(diag['turns']):
                                if turn['speaker'] == 'SYSTEM':
                                    service_call = turn['frames'][0].get('service_call', {})
                                    self.dialogue_id_turn_id_info_map[f"{split}-{diag_id}-{turn_id}"] \
                                        = {'service_call' : service_call}
            with open(json_filename, 'w') as f:
                f.write(json.dumps(self.dialogue_id_turn_id_info_map))

    def tokenize_from_acts(self, dialog_act_list):
        pass # not suitable for template guided, use tokenize_turn to reuse Google's code
    
    def tokenize_turn(self, turn) -> str:
        utterance = self.utter_generator.get_robot_utterance(turn, None)
        return utterance
    
    # override __call__ as we need additional information
    def __call__(self, example, exclude_act=False):
        example['actions'] = example[self.input_field]
        filtered_actions = []
        for action in example['actions']:
            action['act'] = self.act_idx_name_map[action['act']]
            if exclude_act and not action['act'] in {'GOODBYE', 'NOTIFY_FAILURE', 'NOTIFY_SUCCESS', 'REQ_MORE', 'THANK_YOU'}: 
                filtered_actions.append(action)
        if exclude_act:
            example['actions'] = filtered_actions
        split = example['gem_id'].split('-')[-2]
        if split == 'validation':
            split = 'dev'
        diag_turn_key = f"{split}-{example['dialog_id']}-{example['turn_id']}"
        frame_info = self.dialogue_id_turn_id_info_map[diag_turn_key]
        frame_info.update(example)
        turn = {'frames': [frame_info]}
        return {self.output_field: self.tokenize_turn(turn)}
    
    def linearize_by_slots(self, example):
        example['actions'] = example[self.input_field]
        split = example['gem_id'].split('-')[-2]
        if split == 'validation':
            split = 'dev'
        diag_turn_key = f"{split}-{example['dialog_id']}-{example['turn_id']}"
        frame_info = self.dialogue_id_turn_id_info_map[diag_turn_key]
        frame_info.update(example) 
        slot_utter = []
        for action in frame_info['actions']:
            action['act'] = self.act_idx_name_map[action['act']] # convert act id to action name
        all_acts = {action["act"] for action in frame_info["actions"]}
        for action in frame_info["actions"]:
            intent = self.utter_generator._get_intent(action, frame_info)
            utterance = self.utter_generator._get_utterance_for_action(frame_info["service"], intent, action, None)
            if action == 'CONFRIM':
                utterance = "Please confirm the following details: " + utterance
            slot_utter.append((action,utterance))
        return slot_utter

 
class SGD_CopyNaiveLinearizer(SGD_SchemaGuidedLinearizer):
    def __init__(self, act_idx_name_map, schema_paths, separator=' && '):
        super().__init__(act_idx_name_map, schema_paths)
        self.separator = separator
    
    def tokenize_from_acts(self, dialogue_act_list, service):
        res = ''
        for dialog_act in dialogue_act_list:
            act_name = self.act_idx_name_map[dialog_act['act']]
            equal_sign = '='
            if dialog_act['slot'] in self.service_uncategorical_slots[service]:
                equal_sign = 'COPY='
            if dialog_act['values']:
                res += f"{act_name} ( {dialog_act['slot']} {equal_sign} {self.separator.join(dialog_act['values'])} ) "
            elif dialog_act['slot']:
                res += f"{act_name} ( {dialog_act['slot']} ) "
            else:
                res += f"{act_name} "
        return res[:-1] # REMOVE TRAILING SPACE

class SGD_CopySchemaLinearizer(SGD_SchemaGuidedLinearizer):
    def __init__(self, act_idx_name_map, schema_paths, separator=' && '):
        super().__init__(act_idx_name_map, schema_paths)
        self.separator = separator
    
    def tokenize_from_acts(self, dialogue_act_list, service):
        res = ''
        for dialog_act in dialogue_act_list:
            act_name = self.act_idx_name_map[dialog_act['act']]
            equal_sign = '='
            if dialog_act['slot'] in self.service_uncategorical_slots[service]:
                equal_sign = 'COPY='
            slot_desc = self.schema_slots_desc_dict[service][dialog_act['slot']] if not dialog_act['slot'] in {'', 'count', 'intent'} else dialog_act['slot'] # if slot = '' or 'count' or 'intent', keep it as it is
            if dialog_act['values']:
                res += f"{act_name} ( {slot_desc} {equal_sign} {self.separator.join(dialog_act['values'])} ) "
            elif dialog_act['slot']:
                res += f"{act_name} ( {slot_desc} ) "
            else:
                res += f"{act_name} "
        return res[:-1] # REMOVE TRAILING SPACE
