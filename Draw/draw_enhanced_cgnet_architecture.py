#!/usr/bin/env python3
"""
Enhanced CGNet Architecture Diagram with User Modifications
Highlights the specific modifications made to the base CGNet architecture:
1. ASPP Module with multiple dilation rates
2. ResNet34 backbone support 
3. Enhanced Change Guiding Module
4. Improved decoder with better feature fusion
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_enhanced_cgnet_diagram():
    # Set up the figure with better layout
    fig, ax = plt.subplots(1, 1, figsize=(20, 14))
    
    # Color scheme for modifications
    colors = {
        'backbone': '#E8F4FD',
        'original': '#F0F8E8', 
        'modified': '#FFE6E6',  # Red tint for modifications
        'new': '#FF9999',       # Darker red for new components
        'connection': '#666666',
        'text': '#333333'
    }
    
    # Define positions and sizes
    box_height = 0.8
    box_width = 2.5
    spacing_x = 3.5
    spacing_y = 2.0
    
    # Starting positions
    start_x = 1
    start_y = 10
    
    # Function to create boxes with custom styling
    def create_box(x, y, width, height, text, color, is_modified=False, is_new=False):
        if is_new:
            box_color = colors['new']
            edge_color = '#CC0000'
            edge_width = 3
        elif is_modified:
            box_color = colors['modified']
            edge_color = '#FF6666'
            edge_width = 2
        else:
            box_color = color
            edge_color = '#888888'
            edge_width = 1
            
        box = FancyBboxPatch(
            (x, y), width, height,
            boxstyle="round,pad=0.1",
            facecolor=box_color,
            edgecolor=edge_color,
            linewidth=edge_width
        )
        ax.add_patch(box)
        
        # Add text with better formatting
        ax.text(x + width/2, y + height/2, text, 
               ha='center', va='center', fontsize=9, fontweight='bold',
               color=colors['text'], wrap=True)
        
        return box
    
    # Function to create arrows
    def create_arrow(start_pos, end_pos, color='black', style='->', width=1.5):
        arrow = ConnectionPatch(start_pos, end_pos, "data", "data",
                              arrowstyle=style, shrinkA=5, shrinkB=5,
                              mutation_scale=20, fc=color, ec=color, lw=width)
        ax.add_patch(arrow)
    
    # Title
    ax.text(10, 12.5, 'Enhanced CGNet Architecture with User Modifications', 
           fontsize=16, fontweight='bold', ha='center')
    
    # Legend
    legend_y = 11.8
    create_box(1, legend_y, 1.8, 0.4, 'Original\nComponents', colors['original'])
    create_box(3.2, legend_y, 1.8, 0.4, 'Modified\nComponents', colors['modified'], is_modified=True)
    create_box(5.4, legend_y, 1.8, 0.4, 'New\nComponents', colors['new'], is_new=True)
    
    # Input Images
    img_a = create_box(start_x, start_y, box_width, box_height, 'Image A\n(T1)', colors['backbone'])
    img_b = create_box(start_x + spacing_x, start_y, box_width, box_height, 'Image B\n(T2)', colors['backbone'])
    
    # Backbone Networks (Modified to support ResNet34)
    backbone_y = start_y - spacing_y
    backbone_a = create_box(start_x, backbone_y, box_width, box_height, 
                           'ResNet34/VGG16\nBackbone A', colors['backbone'], is_modified=True)
    backbone_b = create_box(start_x + spacing_x, backbone_y, box_width, box_height, 
                           'ResNet34/VGG16\nBackbone B', colors['backbone'], is_modified=True)
    
    # Feature extraction levels
    levels = ['Level 1\n(64 ch)', 'Level 2\n(128 ch)', 'Level 3\n(256 ch)', 'Level 4\n(512 ch)']
    feature_y = backbone_y - spacing_y
    
    # Features A
    features_a = []
    for i, level in enumerate(levels):
        feat = create_box(start_x, feature_y - i*1.2, box_width*0.8, box_height*0.7, 
                         f'Feat A\n{level}', colors['original'])
        features_a.append(feat)
    
    # Features B  
    features_b = []
    for i, level in enumerate(levels):
        feat = create_box(start_x + spacing_x, feature_y - i*1.2, box_width*0.8, box_height*0.7, 
                         f'Feat B\n{level}', colors['original'])
        features_b.append(feat)
    
    # Change Guiding Module (Enhanced)
    cgm_x = start_x + spacing_x * 2
    cgm = create_box(cgm_x, feature_y - 1.5, box_width, box_height*1.5, 
                    'Enhanced Change\nGuiding Module\n(CGM)\n\nBilinear\nInterpolation +\nElement-wise\nOperations', 
                    colors['modified'], is_modified=True)
    
    # ASPP Module (New Addition)
    aspp_y = feature_y - 3.5
    aspp = create_box(cgm_x, aspp_y, box_width, box_height*1.2, 
                     'ASPP Module\n(NEW)\n\nDilations:\n1, 4, 8, 12, 24\nGELU Activation', 
                     colors['new'], is_new=True)
    
    # Enhanced Decoder
    decoder_x = start_x + spacing_x * 3
    decoder_blocks = []
    decoder_y_start = feature_y
    
    for i in range(4):
        decoder_name = f'Enhanced\nDecoder {4-i}\n({512//(2**i)} ch)'
        if i == 0:  # First decoder block has ASPP integration
            decoder_name += '\n+ ASPP\nIntegration'
        
        decoder = create_box(decoder_x, decoder_y_start - i*1.2, box_width, box_height*0.9, 
                           decoder_name, colors['modified'], is_modified=True)
        decoder_blocks.append(decoder)
    
    # Final prediction layers
    pred_x = start_x + spacing_x * 4.5
    pred_y = feature_y - 1
    
    pred1 = create_box(pred_x, pred_y, box_width, box_height, 
                      'Prediction 1\n(Intermediate)', colors['original'])
    pred2 = create_box(pred_x, pred_y - 1.5, box_width, box_height, 
                      'Prediction 2\n(Final)', colors['original'])
    
    # Loss computation (Enhanced)
    loss_x = pred_x
    loss_y = pred_y - 3.5
    loss_box = create_box(loss_x, loss_y, box_width, box_height*1.5, 
                         'Enhanced Loss\n\nBCE + Dice +\nFocal + Boundary\n+ Label Smoothing', 
                         colors['modified'], is_modified=True)
    
    # Add connections
    # Input to backbone
    create_arrow((start_x + box_width/2, start_y), 
                (start_x + box_width/2, backbone_y + box_height))
    create_arrow((start_x + spacing_x + box_width/2, start_y), 
                (start_x + spacing_x + box_width/2, backbone_y + box_height))
    
    # Backbone to features
    for i in range(4):
        # Features A
        create_arrow((start_x + box_width/2, backbone_y), 
                    (start_x + box_width*0.4, feature_y - i*1.2 + box_height*0.35))
        # Features B
        create_arrow((start_x + spacing_x + box_width/2, backbone_y), 
                    (start_x + spacing_x + box_width*0.4, feature_y - i*1.2 + box_height*0.35))
        
        # Features to CGM
        create_arrow((start_x + box_width*0.8, feature_y - i*1.2 + box_height*0.35), 
                    (cgm_x, feature_y - 1.5 + box_height*0.75))
        create_arrow((start_x + spacing_x + box_width*0.8, feature_y - i*1.2 + box_height*0.35), 
                    (cgm_x, feature_y - 1.5 + box_height*0.75))
    
    # CGM to ASPP
    create_arrow((cgm_x + box_width/2, feature_y - 1.5), 
                (cgm_x + box_width/2, aspp_y + box_height*1.2), 
                color='red', width=2)
    
    # ASPP to Decoder
    create_arrow((cgm_x + box_width, aspp_y + box_height*0.6), 
                (decoder_x, decoder_y_start + box_height*0.45), 
                color='red', width=2)
    
    # Decoder connections
    for i in range(3):
        create_arrow((decoder_x + box_width/2, decoder_y_start - i*1.2), 
                    (decoder_x + box_width/2, decoder_y_start - (i+1)*1.2 + box_height*0.9))
    
    # Decoder to predictions
    create_arrow((decoder_x + box_width, decoder_y_start - 1.2*1.5), 
                (pred_x, pred_y + box_height/2))
    create_arrow((decoder_x + box_width, decoder_y_start - 1.2*3), 
                (pred_x, pred_y - 1.5 + box_height/2))
    
    # Predictions to loss
    create_arrow((pred_x + box_width/2, pred_y), 
                (pred_x + box_width/2, loss_y + box_height*1.5), 
                color='purple', width=2)
    create_arrow((pred_x + box_width/2, pred_y - 1.5), 
                (pred_x + box_width/2, loss_y + box_height*1.5), 
                color='purple', width=2)
    
    # Add modification annotations
    ax.text(cgm_x + box_width + 0.3, feature_y - 1.5, 
           'ENHANCED:\nBetter feature\nfusion and\ninterpolation', 
           fontsize=8, color='red', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE6E6'))
    
    ax.text(cgm_x + box_width + 0.3, aspp_y, 
           'NEW:\nMulti-scale\ncontext with\nASPP module', 
           fontsize=8, color='darkred', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#FF9999'))
    
    ax.text(decoder_x + box_width + 0.3, decoder_y_start - 2, 
           'ENHANCED:\nImproved decoder\nwith ASPP\nintegration', 
           fontsize=8, color='red', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE6E6'))
    
    ax.text(pred_x + box_width + 0.3, loss_y, 
           'ENHANCED:\nComposite loss\nwith boundary\nand smoothing', 
           fontsize=8, color='red', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE6E6'))
    
    # Add backbone modification note
    ax.text(start_x + spacing_x*2 - 1, backbone_y + 1, 
           'MODIFIED:\nAdded ResNet34\nsupport alongside\noriginal VGG16', 
           fontsize=8, color='red', fontweight='bold',
           bbox=dict(boxstyle="round,pad=0.3", facecolor='#FFE6E6'))
    
    # Set axis properties
    ax.set_xlim(0, 20)
    ax.set_ylim(0, 13)
    ax.axis('off')
    
    plt.tight_layout()
    return fig

def save_diagram():
    """Save the architecture diagram"""
    fig = create_enhanced_cgnet_diagram()
    
    # Save in multiple formats
    output_path = "enhanced_cgnet_architecture_with_modifications"
    fig.savefig(f"{output_path}.png", dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    fig.savefig(f"{output_path}.pdf", bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    plt.show()
    
    print(f"Enhanced CGNet architecture diagram saved as:")
    print(f"- {output_path}.png (high resolution)")
    print(f"- {output_path}.pdf (vector format)")

if __name__ == "__main__":
    print("Creating Enhanced CGNet Architecture Diagram...")
    print("This diagram highlights your specific modifications to the base CGNet:")
    print("1. ASPP Module with multiple dilation rates (1,4,8,12,24)")
    print("2. ResNet34 backbone support")  
    print("3. Enhanced Change Guiding Module")
    print("4. Improved decoder with ASPP integration")
    print("5. Composite loss function with boundary loss and label smoothing")
    print()
    
    save_diagram()
