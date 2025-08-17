from graphviz import Digraph

def draw_cgnet_internal_architecture():
    dot = Digraph(comment='CGNet Internal Architecture with ResNet34', format='png')
    dot.attr(rankdir='TB', splines='ortho', nodesep='0.2', ranksep='0.5')
    dot.attr('node', shape='box', style='rounded,filled', fontname='Helvetica', fontsize='10')
    dot.attr('edge', fontname='Helvetica', fontsize='9')

    # --- Main Graph ---
    with dot.subgraph(name='cluster_main') as c:
        c.attr(label='CGNet Internal Architecture (ResNet34 Backbone)', fontsize='14', fontname='Helvetica-Bold')
        
        # --- Inputs ---
        with c.subgraph(name='cluster_inputs') as inputs:
            inputs.attr(label='Inputs', style='invis')
            inputs.node('A', 'Image A\n(3, 256, 256)', fillcolor='lightblue')
            inputs.node('B', 'Image B\n(3, 256, 256)', fillcolor='lightblue')

        # --- Siamese Encoders (ResNet34) ---
        with c.subgraph(name='cluster_encoder') as encoder:
            encoder.attr(label='Siamese ResNet34 Encoders', style='filled', color='lightgrey')
            
            # Backbone A
            with encoder.subgraph(name='cluster_backbone_A') as bb_a:
                bb_a.attr(label='Backbone A', style='rounded')
                bb_a.node('inc_A', 'Inc (Conv+BN+ReLU+Pool)\n[64, 128, 128]', fillcolor='white')
                bb_a.node('d1_A', 'Layer1\n[64, 64, 64]', fillcolor='white')
                bb_a.node('d2_A', 'Layer2\n[128, 32, 32]', fillcolor='white')
                bb_a.node('d3_A', 'Layer3\n[256, 16, 16]', fillcolor='white')
                bb_a.node('d4_A', 'Layer4\n[512, 8, 8]', fillcolor='white')
                bb_a.edge('inc_A', 'd1_A')
                bb_a.edge('d1_A', 'd2_A')
                bb_a.edge('d2_A', 'd3_A')
                bb_a.edge('d3_A', 'd4_A')

            # Backbone B
            with encoder.subgraph(name='cluster_backbone_B') as bb_b:
                bb_b.attr(label='Backbone B', style='rounded')
                bb_b.node('inc_B', 'Inc (Conv+BN+ReLU+Pool)\n[64, 128, 128]', fillcolor='white')
                bb_b.node('d1_B', 'Layer1\n[64, 64, 64]', fillcolor='white')
                bb_b.node('d2_B', 'Layer2\n[128, 32, 32]', fillcolor='white')
                bb_b.node('d3_B', 'Layer3\n[256, 16, 16]', fillcolor='white')
                bb_b.node('d4_B', 'Layer4\n[512, 8, 8]', fillcolor='white')
                bb_b.edge('inc_B', 'd1_B')
                bb_b.edge('d1_B', 'd2_B')
                bb_b.edge('d2_B', 'd3_B')
                bb_b.edge('d3_B', 'd4_B')

        # --- Feature Fusion ---
        with c.subgraph(name='cluster_fusion') as fusion:
            fusion.attr(label='Feature Fusion', style='filled', color='lightgrey')
            fusion.node('cat1', 'Concat\n[128, 64, 64]', shape='ellipse', fillcolor='lemonchiffon')
            fusion.node('cat2', 'Concat\n[256, 32, 32]', shape='ellipse', fillcolor='lemonchiffon')
            fusion.node('cat3', 'Concat\n[512, 16, 16]', shape='ellipse', fillcolor='lemonchiffon')
            fusion.node('cat4', 'Concat\n[1024, 8, 8]', shape='ellipse', fillcolor='lemonchiffon')
            
            fusion.node('red1', 'ConvReduce 1\n[64, 64, 64]', fillcolor='lightyellow')
            fusion.node('red2', 'ConvReduce 2\n[128, 32, 32]', fillcolor='lightyellow')
            fusion.node('red3', 'ConvReduce 3\n[256, 16, 16]', fillcolor='lightyellow')
            fusion.node('red4', 'ConvReduce 4\n[512, 8, 8]', fillcolor='lightyellow')
            
            fusion.edge('cat1', 'red1')
            fusion.edge('cat2', 'red2')
            fusion.edge('cat3', 'red3')
            fusion.edge('cat4', 'red4')

        # --- Change Prior Path ---
        with c.subgraph(name='cluster_prior') as prior:
            prior.attr(label='Path 1: Generate Change Prior', style='filled', color='lightcyan')
            prior.node('aspp', 'ASPP Module (Optional)', fillcolor='white')
            prior.node('prior_decoder', 'Simple Decoder\n(Conv -> Conv)', fillcolor='white')
            prior.node('change_map_prior', 'Initial Change Map\n(The "Change Prior")', shape='note', fillcolor='lightpink')
            prior.edge('aspp', 'prior_decoder')
            prior.edge('prior_decoder', 'change_map_prior')

        # --- Guided Decoder Path ---
        with c.subgraph(name='cluster_decoder') as decoder:
            decoder.attr(label='Path 2: Guided Decoder', style='filled', color='lightcyan')
            decoder.node('cgm4', 'ChangeGuideModule 4', shape='parallelogram', fillcolor='lightpink')
            decoder.node('dec_mod4', 'DecoderModule 4\n(Upsample -> Concat -> Conv)\n[256, 16, 16]', fillcolor='lightgreen')
            
            decoder.node('cgm3', 'ChangeGuideModule 3', shape='parallelogram', fillcolor='lightpink')
            decoder.node('dec_mod3', 'DecoderModule 3\n(Upsample -> Concat -> Conv)\n[128, 32, 32]', fillcolor='lightgreen')

            decoder.node('cgm2', 'ChangeGuideModule 2', shape='parallelogram', fillcolor='lightpink')
            decoder.node('dec_mod2', 'DecoderModule 2\n(Upsample -> Concat -> Conv)\n[64, 64, 64]', fillcolor='lightgreen')
            
            decoder.node('final_decoder', 'Final Decoder\n(Conv -> Upsample)', fillcolor='lightgreen')
            decoder.node('final_map', 'Final Change Map', shape='note', fillcolor='lightpink')

            decoder.edge('cgm4', 'dec_mod4')
            decoder.edge('dec_mod4', 'cgm3')
            decoder.edge('cgm3', 'dec_mod3')
            decoder.edge('dec_mod3', 'cgm2')
            decoder.edge('cgm2', 'dec_mod2')
            decoder.edge('dec_mod2', 'final_decoder')
            decoder.edge('final_decoder', 'final_map')

    # --- Connections ---
    # Inputs to Backbones
    dot.edge('A', 'inc_A')
    dot.edge('B', 'inc_B')

    # Backbone to Fusion
    dot.edge('d1_A', 'cat1', style='dashed')
    dot.edge('d1_B', 'cat1', style='dashed')
    dot.edge('d2_A', 'cat2', style='dashed')
    dot.edge('d2_B', 'cat2', style='dashed')
    dot.edge('d3_A', 'cat3', style='dashed')
    dot.edge('d3_B', 'cat3', style='dashed')
    dot.edge('d4_A', 'cat4', style='dashed')
    dot.edge('d4_B', 'cat4', style='dashed')

    # Fusion to Decoder and Prior Path
    dot.edge('red4', 'aspp')
    dot.edge('red4', 'cgm4')
    
    # Prior to Guided Path
    dot.edge('change_map_prior', 'cgm4', style='dashed', constraint='false')
    dot.edge('change_map_prior', 'cgm3', style='dashed', constraint='false')
    dot.edge('change_map_prior', 'cgm2', style='dashed', constraint='false')

    # Skip connections from Encoder to Decoder
    dot.edge('red3', 'dec_mod4', style='dashed')
    dot.edge('red2', 'dec_mod3', style='dashed')
    dot.edge('red1', 'dec_mod2', style='dashed')

    # Render and Save
    try:
        dot.render('cgnet_internal_architecture', view=True)
        print("Internal architecture diagram saved as cgnet_internal_architecture.png")
    except Exception as e:
        print(f"Error rendering graph: {e}")
        print("Please ensure you have Graphviz installed and in your system's PATH.")

if __name__ == '__main__':
    draw_cgnet_internal_architecture()
