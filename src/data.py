#data.py
import json
class ProcessConversationData:
    """
    To see necessary input data structure, please refer to conversation_data.json
    """
    def __init__(self, input_file, users, epochs, method, output_file=None):
        self.input_file = input_file
        self.users = users
        self.epochs = epochs
        self.method = method
        self.output_file = output_file

    def read_json(self, file_path):
        with open(file_path, "r") as json_file:
            return json.load(json_file)

    def verbose_dict(self, data, users):
        """ 
        High-level data structure output:
        { 
            User 0: {SystemMessages: [], Tasks: [], AssistantResponses: [], UserPersona: X, UserFeedback: [], UserSatisfaction/HarmlesnessRatings: [], Text: ""}
            User 1: {SystemMessages: [], Tasks: [], AssistantResponses: [], UserPersona: Y, UserFeedback: [], UserSatisfaction/HarmlesnessRatings: [], Text: ""}
        }

        Data structure with accurate keys:
        {
            User 0: {sys_mes: [], tasks: [], assist_resp: [], user_p: X, user_f: [], user_r: [], text: ""}
            User 1: {sys_mes: [], tasks: [], assist_resp: [], user_p: Y, user_f: [], user_r: [], text: ""}    
        }

        Suggested json file struture: (ask Sam: is this what we want?)
        {
            "SystemMessage": {"response": []}
            "Assistant 0": { "prompt": [], "response": []}
            "User 0": { "prompt": [], "response": [],satisfaction: [], harmlessness: []}
            "Assistant 1": { "prompt": [], "response": []}
            "User 1": { "prompt": [], "response": [],satisfaction: [], harmlessness: []}
            "Assistant 2": { "prompt": [], "response": []}
            "User 2": { "prompt": [], "response": [],satisfaction: [], harmlessness: []}
            "Assistant 3": { "prompt": [], "response": []}
            "User 3": { "prompt": [], "response": [],satisfaction: [], harmlessness: []}
        }


        """
        tasks, responses, personas, feedback, ratings = [], [], [], [], []
        for i, value in enumerate(data.values()):
            # Get the system Message values during the inital run
            if i == 0:
                system_messages = [dic["response"] for dic in value]
            # If you're the assistant data fields, record the tasks and the responses
            elif i & 1:
                tasks.append([dic["prompt"].split("\n\n")[1] for dic in value])
                responses.append([dic["response"] for dic in value])
            # Otherwise, you're on a user dict, get the user feedback and satisfaction ratings
            else:
                personas.append(value[0]["prompt"].split("System: Please adopt the following persona for all your responses:")[1])
                feedback.append([dic["response"] for dic in value])
                ratings.append([(dic["Satisfaction"], dic["Harmlessness (community)"]) for dic in value])
        # Create the verbose_dict using the lists 
        verbose_dict = {i: {
                "sys_mes": system_messages,
                "tasks": tasks[i],
                "assist_resp": responses[i],
                "user_p": personas[i],
                "user_f": feedback[i],
                "user_r": ratings[i],
                "text": self.create_conversation_text(responses[i], feedback[i])
            } for i in range(users)}

        return verbose_dict

    # Concatenate the responses and feedbacks from the lists for each user
    def create_conversation_text(self, responses, feedback):
        return ' '.join([f'Response: {r} Feedback: {f}' for r, f in zip(responses, feedback)])

    def conv_dict_with_data(self, responses, feedback, users, tasks):
        # Create the dictionary that holds just the text conversation data
        succint_dict = {}
        for i in range(users):
            succint_dict["User " + str(i)] = [self.create_conversation_text(responses[i], feedback[i]), tasks[i]]
        return succint_dict

    def conv_dict(self, data, users):
        responses = []
        tasks = []
        feedback = []
        for i, value in enumerate(data.values()):
            # Skip the system message on the initial run
            if i == 0:
                continue
            # if you're on an assistant dict, get the assistant responses
            elif i & 1:
                responses.append([dic["response"] for dic in value])
                tasks.append(value[0]["prompt"].split("\n\n")[1])
            # otherwise, you're on a user dict, get the user feedback
            else:
                feedback.append([dic["response"] for dic in value])

        return self.conv_dict_with_data(responses, feedback, users, tasks)

    def write_to_text_file(self, filepath, data_dict):
        with open(filepath, 'w') as f:
            for key, value in data_dict.items():
                interaction = f"{key} interaction with Assistant:\n{value[1]} {value[0]}\n<end of text>\n"
                f.write(interaction)

    def write_to_json_file(self, filepath, data_dict):
        json_dict = {}
        with open(filepath, 'w') as f:
            for i, v in data_dict.items():
                json_dict[i] = f"User interaction with Assistant:\n{v[1]} {v[0]}\n<end of text>\n"
            json.dump(json_dict, f)

    def main(self):
        # Argument line: data-file | num_users | epochs | method | optional_file_name (if method is -c)
        # Method is -v for verbose dictionary (all of the information) or -c for only text conversations
        # If -c is used as a method, it will write into the file specified at the 5th argument
        data = self.read_json(self.input_file)
        if self.method == "-v":
            print(self.verbose_dict(data, self.users))
        else:
            data_dict = self.conv_dict(data, self.users)
            self.write_to_json_file(f'{self.output_file}', data_dict)
            













