from graphviz import Digraph

def draw_project_workflow():
    dot = Digraph(comment='CGNet Project Workflow', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.5', ranksep='1.2', dpi='150')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='12')
    dot.attr('edge', fontname='Helvetica', fontsize='10')

    # --- 1. Data Preparation Phase ---
    with dot.subgraph(name='cluster_prep') as c:
        c.attr(label='Phase 1: Data Preparation', style='filled', color='lightgrey', fontsize='14', fontname='Helvetica-Bold')
        c.node('raw_data', 'Raw Datasets\n(e.g., LEVIR-CD, WHU-CD)', shape='folder', fillcolor='khaki', height='0.8')
        c.node('preprocess', '`preprocess_dataset.py`\n\n- Crop large images into 256x256 patches', fillcolor='lightblue2', height='0.8')
        c.node('patched_data', 'Patched Dataset\n(train/val/test splits)', shape='folder', fillcolor='khaki', height='0.8')
        c.edge('raw_data', 'preprocess')
        c.edge('preprocess', 'patched_data')

    # --- 2. Training Phase ---
    with dot.subgraph(name='cluster_train') as c:
        c.attr(label='Phase 2: Model Training', style='filled', color='lightgrey', fontsize='14', fontname='Helvetica-Bold')
        c.node('train_script', '`train_CGNet.py`', fillcolor='lightblue2', width='3.5', height='0.8')
        
        with c.subgraph(name='cluster_train_setup') as setup:
            setup.attr(label='Interactive Setup', style='rounded', color='darkseagreen', fontsize='12')
            setup.node('select_ds', '1. Select Dataset', fillcolor='white')
            setup.node('select_bb', '2. Select Backbone\n(VGG16/ResNet34)', fillcolor='white')
            setup.node('select_aspp', '3. Enable ASPP? (Y/N)', fillcolor='white')
            setup.node('select_bs', '4. Set Batch Size', fillcolor='white')

        with c.subgraph(name='cluster_train_loop') as loop:
            loop.attr(label='Training & Validation Loop', style='rounded', color='darkseagreen', fontsize='12')
            loop.node('model', 'Instantiate CGNet Model', fillcolor='white')
            loop.node('loss', 'Composite Loss\n- BCE + Dice + Focal + Boundary', shape='ellipse', fillcolor='lightpink')
            loop.node('optimizer', 'AdamW Optimizer &\nCosineAnnealingLR Scheduler', fillcolor='white')
            loop.node('validation', 'Validate on Val Set\n(Calculate IoU, F1, etc.)', fillcolor='white')
            loop.node('save_model', 'Save Best Model\n(based on validation IoU)', shape='cylinder', fillcolor='mediumpurple1')
            loop.node('logging', 'Log to TensorBoard', shape='note', fillcolor='lemonchiffon')
            
            loop.edge('model', 'loss')
            loop.edge('loss', 'optimizer')
            loop.edge('optimizer', 'validation')
            loop.edge('validation', 'save_model', label=' if new best IoU')
            loop.edge('validation', 'logging')

        c.edge('select_ds', 'model')
        c.edge('select_bb', 'model')
        c.edge('select_aspp', 'model')

    # --- 3. Testing Phase ---
    with dot.subgraph(name='cluster_test') as c:
        c.attr(label='Phase 3: Evaluation', style='filled', color='lightgrey', fontsize='14', fontname='Helvetica-Bold')
        c.node('test_script', '`test.py`', fillcolor='lightblue2', width='3.5', height='0.8')
        c.node('load_model', 'Load Trained Weights\n(_best_iou.pth)', shape='cylinder', fillcolor='mediumpurple1')
        
        with c.subgraph(name='cluster_test_process') as p:
            p.attr(label='Inference & Post-Processing', style='rounded', color='darkseagreen', fontsize='12')
            p.node('tta', 'Test-Time Augmentation (TTA)\n(Original + Flips)', fillcolor='white')
            p.node('threshold', "Otsu's Thresholding\n(Convert probability to binary mask)", fillcolor='white')
            p.node('morph', 'Morphological Operations\n(Clean up noise)', fillcolor='white')
            p.node('filter', 'Connected Component Filtering\n(Remove small objects)', fillcolor='white')

        c.node('eval_metrics', 'Calculate Final Metrics\n(F1, IoU, Precision, Recall, Kappa)', shape='ellipse', fillcolor='lightpink')
        c.node('final_maps', 'Save Final Change Masks (.png)', shape='folder', fillcolor='khaki')

        c.edge('load_model', 'tta')
        c.edge('tta', 'threshold')
        c.edge('threshold', 'morph')
        c.edge('morph', 'filter')
        c.edge('filter', 'eval_metrics')
        c.edge('filter', 'final_maps')

    # --- Connect Phases ---
    dot.edge('patched_data', 'train_script', lhead='cluster_train')
    dot.edge('patched_data', 'test_script', lhead='cluster_test')
    dot.edge('save_model', 'load_model', ltail='cluster_train_loop', lhead='cluster_test')

    # Render and Save
    try:
        dot.render('cgnet_project_workflow_clear', view=True)
        print("Project workflow diagram saved as cgnet_project_workflow_clear.png")
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print("Please ensure you have Graphviz installed and in your system's PATH.")

if __name__ == '__main__':
    draw_project_workflow()