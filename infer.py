from ccmseg import run_inference

run_inference(
    'output/model1.pth',
    'data/set2/images',
    output_dir='set2_infer'
)

run_inference(
    'output/model2.pth',
    'data/set1/images',
    output_dir='set1_infer'
)
