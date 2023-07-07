#data.py
import json
import sys
"""
To see necessary input data structure, please refer to conversation_data.json
"""
def read_json(file_path):
    with open(file_path, "r") as json_file:
        return json.load(json_file)

def verbose_dict(data, users):
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
    """
    tasks, responses, personas, feedback, ratings = [], [], [], [], []
    for i, (_, value) in enumerate(data.items()):
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
            "sys_mes": system_messages[i], 
            "tasks": tasks[i],
            "assist_resp": responses[i],
            "user_p": personas[i],
            "user_f": feedback[i],
            "user_r": ratings[i],
            "text": create_conversation_text(responses[i], feedback[i])
        } for i in range(users)}

    return verbose_dict

# Concatenate the responses and feedbacks from the lists for each user
def create_conversation_text(responses, feedback):
    return ' '.join([f'Response: {r} Feedback: {f}' for r, f in zip(responses, feedback)])

def conv_dict_with_data(responses, feedback, users):
    # Create the dictionary that holds just the text conversation data
    succint_dict = {}
    for i in range(users):
        succint_dict["User " + str(i)] = create_conversation_text(responses[i], feedback[i])
    return succint_dict

def conv_dict(data, users):
    responses = []
    feedback = []
    for i, (_, value) in enumerate(data.items()):
        # Skip the system message on the initial run
        if i == 0:
            continue
        # if you're on an assistant dict, get the assistant responses
        elif i & 1:
            responses.append([dic["response"] for dic in value])
        # otherwise, you're on a user dict, get the user feedback
        else:
            feedback.append([dic["response"] for dic in value])

    return conv_dict_with_data(responses, feedback, users)

def write_to_text_file(filepath, data_dict):
    with open(filepath, 'w') as f:
        f.write('\n<|endoftext|>\n'.join([f'{i} interaction with Assistant\n{v}' for i, v in data_dict.items()]))

def main():
    # Argument line: data-file | num_users | epochs | method | optional_file_name (if method is -c)
    # Method is -v for verbose dictionary (all of the information) or -c for only text conversations
    # If -c is used as a method, it will write into the file specified at the 5th argument
    data = read_json(sys.argv[1])
    if len(sys.argv) < 5:
        raise Exception("Relevant Parameters not listed")
    else:
        arg4 = sys.argv[4]
        if arg4 == "-v":
            print(verbose_dict(data, int(sys.argv[2])))
        elif arg4 == "-c":
            data_dict = conv_dict(data, int(sys.argv[2]))
            write_to_text_file(f'data/{sys.argv[5]}', data_dict)
            return {"text": data_dict}
            

if __name__ == '__main__':
    main()













