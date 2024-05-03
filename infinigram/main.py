import requests

PROMPT = "The circumference of the sun"
state = PROMPT
final_output = PROMPT
DEBUG = False

while True:

    payload = {
        'corpus': 'v4_dolma-v1_6_llama',
        'query_type': 'ntd',
        'query': state,
    }

    result = requests.post('https://api.infini-gram.io/', json=payload).json()

    if not result['result_by_token_id']:
        state = ' '.join(state.split()[1:])
        print("Current state:", state)
        continue

    token_id = next(iter(result['result_by_token_id']))
    
    token_info = result['result_by_token_id'][token_id]
    token = token_info['token']
    if DEBUG:
        print(token)
    if token.isalpha() or "▁" in token:
        if "▁" in token:
            token = token.replace("▁", "")
            state += " " + token
            final_output += " " + token
        else:
            state += token
            final_output += token 
    else:
        state = state + token
        final_output += token
        
    

    if len(final_output.split()) >= 100:
        break
    
    if '<0x0A>' in state or len(state) == 0:
        break
    #print(f"{state=}")
    print(final_output, end='\r')
print()
