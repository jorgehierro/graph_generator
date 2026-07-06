from agent_tool import visualize_neptune_graph
import json 

with open('sample_data_3.json', 'r') as f:
    a = json.load(f)

f.close()


# After querying Neptune, pass the response directly:
image_path = visualize_neptune_graph(
    neptune_response=a,
    output_dir="./output",
    alert_id=5555,
)