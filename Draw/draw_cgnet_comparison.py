#!/usr/bin/env python3
"""
Side-by-Side Comparison: Base CGNet vs Enhanced CGNet
Shows the original architecture and your specific modifications
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import numpy as np

def create_comparison_diagram():
    # Set up the figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 14))
    
    # Color scheme
    colors = {
        'original': '#E8F4FD',
        'modified': '#FFE6E6', 
        'new': '#FF6666',
        'connection': '#666666'
    }
    
    def draw_cgnet_base(ax, title):
        """Draw the base CGNet architecture"""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Basic components
        components = [
            (2, 10, 'Image A', colors['original']),
            (6, 10, 'Image B', colors['original']),
            (2, 8.5, 'VGG16\nBackbone A', colors['original']),
            (6, 8.5, 'VGG16\nBackbone B', colors['original']),
            (4, 6.5, 'Change Guiding\nModule (CGM)', colors['original']),
            (4, 5, 'Basic Decoder', colors['original']),
            (4, 3.5, 'Prediction 1', colors['original']),
            (4, 2, 'Prediction 2', colors['original']),
            (4, 0.5, 'BCE Loss', colors['original'])
        ]
        
        for x, y, text, color in components:
            box = FancyBboxPatch(
                (x-0.8, y-0.3), 1.6, 0.6,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor='#888888',
                linewidth=1
            )
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=9, fontweight='bold')
        
        # Add basic connections
        connections = [
            ((2, 9.7), (2, 8.8)),
            ((6, 9.7), (6, 8.8)),
            ((2, 8.2), (4, 6.8)),
            ((6, 8.2), (4, 6.8)),
            ((4, 6.2), (4, 5.3)),
            ((4, 4.7), (4, 3.8)),
            ((4, 3.2), (4, 2.3)),
            ((4, 1.7), (4, 0.8))
        ]
        
        for start, end in connections:
            ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                    head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax.set_xlim(0, 8)
        ax.set_ylim(0, 11)
        ax.axis('off')
    
    def draw_cgnet_enhanced(ax, title):
        """Draw your enhanced CGNet architecture"""
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        
        # Enhanced components with modification indicators
        components = [
            (2, 10, 'Image A', colors['original'], False),
            (6, 10, 'Image B', colors['original'], False),
            (2, 8.5, 'ResNet34/VGG16\nBackbone A', colors['modified'], True),
            (6, 8.5, 'ResNet34/VGG16\nBackbone B', colors['modified'], True),
            (4, 6.8, 'Enhanced CGM\nBetter Fusion', colors['modified'], True),
            (7, 5.5, 'ASPP Module\nDilations: 1,4,8,12,24', colors['new'], True),
            (4, 4.5, 'Enhanced Decoder\n+ ASPP Integration', colors['modified'], True),
            (4, 3, 'Prediction 1', colors['original'], False),
            (4, 1.5, 'Prediction 2', colors['original'], False),
            (4, 0.2, 'Composite Loss\nBCE+Dice+Focal\n+Boundary+Smoothing', colors['modified'], True)
        ]
        
        for x, y, text, color, is_modified in components:
            edge_color = '#FF6666' if is_modified else '#888888'
            edge_width = 2 if is_modified else 1
            
            box = FancyBboxPatch(
                (x-0.9, y-0.4), 1.8, 0.8,
                boxstyle="round,pad=0.1",
                facecolor=color,
                edgecolor=edge_color,
                linewidth=edge_width
            )
            ax.add_patch(box)
            ax.text(x, y, text, ha='center', va='center', 
                   fontsize=8, fontweight='bold')
        
        # Enhanced connections
        connections = [
            ((2, 9.6), (2, 8.9)),
            ((6, 9.6), (6, 8.9)),
            ((2, 8.1), (4, 7.2)),
            ((6, 8.1), (4, 7.2)),
            ((4, 6.4), (7, 5.9)),  # CGM to ASPP
            ((7, 5.1), (4, 4.9)),  # ASPP to Decoder
            ((4, 4.1), (4, 3.4)),
            ((4, 2.6), (4, 1.9)),
            ((4, 1.1), (4, 0.6))
        ]
        
        for start, end in connections:
            color = 'red' if start[0] == 7 or end[0] == 7 else 'black'
            width = 0.15 if color == 'red' else 0.1
            ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
                    head_width=width, head_length=0.1, fc=color, ec=color, lw=2)
        
        # Add modification labels
        ax.text(8.5, 8.5, 'NEW:\nResNet34\nSupport', fontsize=8, color='red',
               bbox=dict(boxstyle="round", facecolor='#FFE6E6'))
        ax.text(8.5, 6.8, 'ENHANCED:\nBetter Feature\nFusion', fontsize=8, color='red',
               bbox=dict(boxstyle="round", facecolor='#FFE6E6'))
        ax.text(8.5, 5.5, 'NEW:\nMulti-scale\nContext', fontsize=8, color='darkred',
               bbox=dict(boxstyle="round", facecolor='#FF9999'))
        ax.text(8.5, 4.5, 'ENHANCED:\nASPP\nIntegration', fontsize=8, color='red',
               bbox=dict(boxstyle="round", facecolor='#FFE6E6'))
        ax.text(8.5, 0.2, 'ENHANCED:\nComposite\nLoss', fontsize=8, color='red',
               bbox=dict(boxstyle="round", facecolor='#FFE6E6'))
        
        ax.set_xlim(0, 11)
        ax.set_ylim(0, 11)
        ax.axis('off')
    
    # Draw both architectures
    draw_cgnet_base(ax1, "Original CGNet Architecture")
    draw_cgnet_enhanced(ax2, "Your Enhanced CGNet Architecture")
    
    # Add overall title
    fig.suptitle('CGNet Architecture Comparison: Base vs Enhanced Version', 
                fontsize=16, fontweight='bold', y=0.95)
    
    # Add legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['original'], 
                     edgecolor='#888888', label='Original Components'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['modified'], 
                     edgecolor='#FF6666', label='Modified Components'),
        plt.Rectangle((0, 0), 1, 1, facecolor=colors['new'], 
                     edgecolor='#FF6666', label='New Components')
    ]
    fig.legend(handles=legend_elements, loc='upper center', 
              bbox_to_anchor=(0.5, 0.02), ncol=3, fontsize=12)
    
    plt.tight_layout()
    return fig

def create_detailed_modifications_summary():
    """Create a detailed summary of all modifications"""
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.axis('off')
    
    # Title
    ax.text(0.5, 0.95, 'Detailed Summary of CGNet Modifications', 
           fontsize=16, fontweight='bold', ha='center', transform=ax.transAxes)
    
    modifications = [
        ("1. Backbone Enhancement", [
            "• Added ResNet34 support alongside original VGG16",
            "• Improved feature extraction with deeper architecture",
            "• Better gradient flow for training stability"
        ]),
        
        ("2. ASPP Module Integration (NEW)", [
            "• Atrous Spatial Pyramid Pooling with dilations [1,4,8,12,24]",
            "• Multi-scale context capture for better boundaries",
            "• GELU activation for improved non-linearity",
            "• Significant improvement in segmentation accuracy"
        ]),
        
        ("3. Enhanced Change Guiding Module", [
            "• Improved bilinear interpolation strategy", 
            "• Better element-wise feature fusion operations",
            "• More robust change prior computation",
            "• Enhanced spatial attention mechanisms"
        ]),
        
        ("4. Advanced Decoder Architecture", [
            "• Integration with ASPP module outputs",
            "• Progressive feature refinement across scales",
            "• Better skip connections from encoder",
            "• Improved upsampling strategies"
        ]),
        
        ("5. Composite Loss Function", [
            "• BCE Loss with Label Smoothing (α=0.1)",
            "• Dice Loss for better overlap optimization",
            "• Focal Loss (α=0.8, γ=2) for hard example mining",
            "• Boundary Loss for edge preservation",
            "• Weighted combination for balanced training"
        ]),
        
        ("6. Training Enhancements", [
            "• Test-Time Augmentation (TTA) with flips",
            "• Otsu thresholding for optimal threshold selection",
            "• Morphological post-processing for cleaner outputs",
            "• Interactive dataset and model selection",
            "• GPU/CPU automatic fallback mechanism"
        ])
    ]
    
    y_pos = 0.85
    for title, items in modifications:
        # Section title
        ax.text(0.05, y_pos, title, fontsize=12, fontweight='bold', 
               transform=ax.transAxes, color='darkred')
        y_pos -= 0.03
        
        # Items
        for item in items:
            ax.text(0.08, y_pos, item, fontsize=10, 
                   transform=ax.transAxes, color='black')
            y_pos -= 0.025
        
        y_pos -= 0.02  # Extra space between sections
    
    # Performance note
    ax.text(0.05, 0.08, 'Performance Impact:', fontsize=12, fontweight='bold', 
           transform=ax.transAxes, color='darkgreen')
    ax.text(0.08, 0.05, '• LEVIR-CD-256: F1=93.22%, IoU=87.31%', fontsize=10, 
           transform=ax.transAxes, color='darkgreen')
    ax.text(0.08, 0.025, '• Significant improvement over base CGNet', fontsize=10, 
           transform=ax.transAxes, color='darkgreen')
    
    plt.tight_layout()
    return fig

def save_all_diagrams():
    """Save all comparison diagrams"""
    print("Creating comprehensive CGNet modification documentation...")
    
    # 1. Side-by-side comparison
    fig1 = create_comparison_diagram()
    fig1.savefig("cgnet_base_vs_enhanced_comparison.png", dpi=300, bbox_inches='tight')
    fig1.savefig("cgnet_base_vs_enhanced_comparison.pdf", bbox_inches='tight')
    
    # 2. Detailed modifications summary
    fig2 = create_detailed_modifications_summary()
    fig2.savefig("cgnet_modifications_summary.png", dpi=300, bbox_inches='tight')
    fig2.savefig("cgnet_modifications_summary.pdf", bbox_inches='tight')
    
    plt.show()
    
    print("\nDiagrams saved:")
    print("1. cgnet_base_vs_enhanced_comparison.png/pdf - Side-by-side architecture comparison")
    print("2. cgnet_modifications_summary.png/pdf - Detailed modification summary")
    print("\nThese diagrams clearly show your contributions to the base CGNet architecture!")

if __name__ == "__main__":
    save_all_diagrams()
