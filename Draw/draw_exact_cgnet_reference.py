import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, Polygon
import numpy as np

def draw_cgnet_architecture():
    # Create figure with exact proportions matching reference
    fig, ax = plt.subplots(1, 1, figsize=(20, 12))
    
    # Background sections to exactly match reference
    # Upper section (cream/beige)
    upper_bg = Rectangle((0, 0.65), 1, 0.35, facecolor='#F5E6D3', alpha=0.9, transform=ax.transAxes)
    ax.add_patch(upper_bg)
    
    # Middle section (light green)
    middle_bg = Rectangle((0, 0.35), 1, 0.3, facecolor='#E8F5E8', alpha=0.9, transform=ax.transAxes)
    ax.add_patch(middle_bg)
    
    # Lower section (light blue/purple)
    lower_bg = Rectangle((0, 0), 1, 0.35, facecolor='#E8E8FF', alpha=0.9, transform=ax.transAxes)
    ax.add_patch(lower_bg)
    
    # Set axis limits for proper scaling
    ax.set_xlim(0, 24)
    ax.set_ylim(0, 14)
    
    # Title - exact position and style as reference
    ax.text(12, 13.2, 'CGNet', fontsize=28, fontweight='bold', color='red', ha='center')
    
    # Input images (left side) - matching reference exactly
    # First image (forest/green texture)
    img1 = Rectangle((1, 10.5), 2, 2, facecolor='#2E7D32', edgecolor='black', linewidth=2)
    ax.add_patch(img1)
    # Add texture lines to simulate forest
    for i in range(5):
        for j in range(5):
            ax.plot([1.2 + i*0.3, 1.4 + i*0.3], [10.7 + j*0.3, 10.9 + j*0.3], 
                   color='#1B5E20', linewidth=2)
    ax.text(2, 10.2, '256×256×3', fontsize=11, ha='center', va='top', fontweight='bold')
    
    # Second image (urban/brown texture)  
    img2 = Rectangle((1, 7.5), 2, 2, facecolor='#8B4513', edgecolor='black', linewidth=2)
    ax.add_patch(img2)
    # Add grid pattern for urban look
    for i in range(6):
        ax.axvline(x=1.3 + i*0.25, ymin=(7.5)/14, ymax=(9.5)/14, color='#654321', linewidth=1.5)
        ax.axhline(y=7.8 + i*0.25, xmin=1/24, xmax=3/24, color='#654321', linewidth=1.5)
    ax.text(2, 7.2, '256×256×3', fontsize=11, ha='center', va='top', fontweight='bold')
    
    # CNN Backbone blocks - exact 3D perspective as reference
    def draw_3d_cnn_block(x, y, w, h, d, color, label=None):
        # Front face
        front = Rectangle((x, y), w, h, facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(front)
        
        # Top face (parallelogram for 3D effect)
        top_pts = np.array([[x, y+h], [x+w, y+h], [x+w+d, y+h+d], [x+d, y+h+d]])
        top = Polygon(top_pts, facecolor=color, edgecolor='black', linewidth=1, alpha=0.8)
        ax.add_patch(top)
        
        # Right face
        right_pts = np.array([[x+w, y], [x+w+d, y+d], [x+w+d, y+h+d], [x+w, y+h]])
        right = Polygon(right_pts, facecolor=color, edgecolor='black', linewidth=1, alpha=0.6)
        ax.add_patch(right)
        
        if label:
            ax.text(x+w/2, y+h/2, label, fontsize=14, ha='center', va='center', 
                   color='white', fontweight='bold')
    
    # Upper backbone CNN blocks (decreasing size like reference)
    cnn_sizes = [(1.8, 1.4), (1.6, 1.2), (1.4, 1.0), (1.2, 0.8)]
    cnn_x_positions = [4.5, 6.8, 9.0, 11.1]
    
    for i, ((w, h), x) in enumerate(zip(cnn_sizes, cnn_x_positions)):
        draw_3d_cnn_block(x, 11, w, h, 0.3, '#1976D2', str(i+1) if i < 4 else '')
    
    # Lower backbone CNN blocks
    for i, ((w, h), x) in enumerate(zip(cnn_sizes, cnn_x_positions)):
        draw_3d_cnn_block(x, 8, w, h, 0.3, '#1976D2', str(i+1) if i < 4 else '')
    
    # Change Guide Modules (CGM) - exact purple blocks as reference with angled flow
    cgm_positions = [(13.5, 11.2), (16.0, 10.8), (18.5, 10.4), (21.0, 10.0)]
    for i, (x, y) in enumerate(cgm_positions):
        cgm = Rectangle((x, y), 1.8, 0.8, facecolor='#9C27B0', edgecolor='black', linewidth=1.5)
        ax.add_patch(cgm)
        ax.text(x + 0.9, y + 0.4, str(i+1), fontsize=14, ha='center', va='center', 
               color='white', fontweight='bold')
        
        # Addition circles before CGMs
        circle = Circle((x-0.8, y+0.4), 0.25, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x-0.8, y+0.4, '+', fontsize=16, ha='center', va='center', fontweight='bold')
    
    # Prediction output (top right) - exact black square
    pred_box = Rectangle((22, 10.8), 1.5, 1.5, facecolor='black', edgecolor='black', linewidth=2)
    ax.add_patch(pred_box)
    # White squares inside to show change map pattern
    for i in range(3):
        for j in range(3):
            if (i+j) % 2 == 0:  # Checkerboard pattern
                mini_sq = Rectangle((22.2 + i*0.35, 11.0 + j*0.35), 0.3, 0.3, 
                                  facecolor='white', edgecolor='none')
                ax.add_patch(mini_sq)
    ax.text(22.75, 12.5, 'Prediction', fontsize=12, ha='center', va='bottom', fontweight='bold')
    
    # Main flow arrows (black horizontal)
    arrow_props = dict(arrowstyle='->', lw=2.5, color='black')
    
    # Flow from CNN blocks to CGMs
    ax.annotate('', xy=(12.7, 11.6), xytext=(11.1+1.2, 11.6), arrowprops=arrow_props)
    
    # Between CGMs (angled to match reference)
    for i in range(len(cgm_positions)-1):
        start_x = cgm_positions[i][0] + 1.8
        start_y = cgm_positions[i][1] + 0.4
        end_x = cgm_positions[i+1][0] - 0.8
        end_y = cgm_positions[i+1][1] + 0.4
        ax.annotate('', xy=(end_x, end_y), xytext=(start_x, start_y), arrowprops=arrow_props)
    
    # Final arrow to prediction
    ax.annotate('', xy=(22, 11.5), xytext=(21+1.8, 10.4), arrowprops=arrow_props)
    
    # Vertical skip connections (orange/yellow) - exact as reference
    skip_props = dict(arrowstyle='->', lw=4, color='#FF9800')
    skip_x_positions = [14.4, 16.9, 19.4]
    
    for x in skip_x_positions:
        ax.annotate('', xy=(x, 6.8), xytext=(x, 10.2), arrowprops=skip_props)
    
    # Middle section deep feature extraction
    # Deep Feature block with texture
    deep_feat = Rectangle((4, 5.8), 2.5, 1.2, facecolor='#4A148C', edgecolor='black', linewidth=1.5)
    ax.add_patch(deep_feat)
    # Add pattern inside deep feature
    for i in range(4):
        ax.plot([4.3 + i*0.5, 4.3 + i*0.5], [6.0, 6.8], color='#E1BEE7', linewidth=2)
    ax.text(5.25, 6.4, 'Deep Feature', fontsize=11, ha='center', va='center', 
           color='white', fontweight='bold')
    
    # Change Map with pattern
    change_map = Rectangle((7, 5.8), 2.5, 1.2, facecolor='#2E7D32', edgecolor='black', linewidth=1.5)
    ax.add_patch(change_map)
    # Binary pattern for change map
    for i in range(5):
        for j in range(3):
            if (i+j) % 2 == 0:
                mini_rect = Rectangle((7.2 + i*0.4, 6.0 + j*0.25), 0.3, 0.2, 
                                    facecolor='white', edgecolor='none')
                ax.add_patch(mini_rect)
    ax.text(8.25, 6.4, 'Change Map', fontsize=11, ha='center', va='center', 
           color='white', fontweight='bold')
    
    # Lower flow circles and processing blocks
    lower_x_positions = [14, 16.5, 19, 21.5]
    
    for i, x in enumerate(lower_x_positions):
        # Addition circles
        circle = Circle((x, 6.2), 0.3, facecolor='white', edgecolor='black', linewidth=2)
        ax.add_patch(circle)
        ax.text(x, 6.2, '+', fontsize=18, ha='center', va='center', fontweight='bold')
        
        # Processing blocks with 3D effect (decreasing numbers 4,3,2)
        if i < 3:
            # Main block
            block = Rectangle((x-0.8, 4.8), 1.6, 0.8, facecolor='#D32F2F', edgecolor='black', linewidth=1.5)
            ax.add_patch(block)
            # 3D effect
            top_3d = Polygon([[x-0.8, 5.6], [x+0.8, 5.6], [x+1.0, 5.8], [x-0.6, 5.8]], 
                           facecolor='#F44336', edgecolor='black', linewidth=1)
            ax.add_patch(top_3d)
            right_3d = Polygon([[x+0.8, 4.8], [x+1.0, 5.0], [x+1.0, 5.8], [x+0.8, 5.6]], 
                             facecolor='#C62828', edgecolor='black', linewidth=1)
            ax.add_patch(right_3d)
            
            ax.text(x, 5.2, str(4-i), fontsize=14, ha='center', va='center', 
                   color='white', fontweight='bold')
    
    # Blue horizontal arrows in middle section
    blue_arrow_props = dict(arrowstyle='->', lw=3, color='#2196F3')
    for i in range(3):
        ax.annotate('', xy=(lower_x_positions[i+1], 6.2), xytext=(lower_x_positions[i], 6.2), 
                   arrowprops=blue_arrow_props)
    
    # Classifier with 3D effect
    classifier = Rectangle((20.7, 5.8), 2, 1, facecolor='#4CAF50', edgecolor='black', linewidth=1.5)
    ax.add_patch(classifier)
    # 3D top
    class_top = Polygon([[20.7, 6.8], [22.7, 6.8], [22.9, 7.0], [20.9, 7.0]], 
                       facecolor='#66BB6A', edgecolor='black', linewidth=1)
    ax.add_patch(class_top)
    ax.text(21.7, 6.3, 'Classifier', fontsize=12, ha='center', va='center', 
           color='white', fontweight='bold')
    
    # Lower section component detail boxes
    # VGG16_BN with enhanced styling
    vgg_box = Rectangle((9, 0.8), 4, 2, facecolor='#B0BEC5', edgecolor='black', linewidth=1.5)
    ax.add_patch(vgg_box)
    ax.text(11, 2.4, 'VGG16_BN', fontsize=13, ha='center', va='center', fontweight='bold')
    ax.text(11, 1.9, 'Block Type:', fontsize=11, ha='center', va='center')
    ax.text(11, 1.4, '1,2,3,4', fontsize=11, ha='center', va='center', fontweight='bold')
    
    # CGM Module with detailed styling
    cgm_detail = Rectangle((13.5, 0.8), 4, 2, facecolor='#FFAB91', edgecolor='black', linewidth=1.5)
    ax.add_patch(cgm_detail)
    ax.text(15.5, 2.1, 'CGM: Change', fontsize=12, ha='center', va='center', fontweight='bold')
    ax.text(15.5, 1.7, 'Guide Module', fontsize=12, ha='center', va='center', fontweight='bold')
    
    # Convolutional Block with symbols
    conv_detail = Rectangle((18, 0.8), 4.5, 2, facecolor='#CE93D8', edgecolor='black', linewidth=1.5)
    ax.add_patch(conv_detail)
    ax.text(20.25, 2.4, 'Convolutional Block', fontsize=11, ha='center', va='center', fontweight='bold')
    ax.text(20.25, 2.0, '⊗: Convolutional Layer', fontsize=9, ha='center', va='center')
    ax.text(20.25, 1.6, '⊕: Batch Normalization', fontsize=9, ha='center', va='center')
    ax.text(20.25, 1.2, '◯: Activation', fontsize=9, ha='center', va='center')
    
    # Legend with exact styling
    legend_props = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='black', linewidth=1)
    ax.text(1, 1.5, '→ Next Step', fontsize=11, ha='left', va='center', bbox=legend_props)
    
    blue_legend_props = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#2196F3', linewidth=1)
    ax.text(1, 1, '→ Upsampling', fontsize=11, ha='left', va='center', color='#2196F3', bbox=blue_legend_props)
    
    orange_legend_props = dict(boxstyle="round,pad=0.3", facecolor='white', edgecolor='#FF9800', linewidth=1)
    ax.text(1, 0.5, '→ Resize', fontsize=11, ha='left', va='center', color='#FF9800', bbox=orange_legend_props)
    
    # Connecting dashed lines from detail boxes to processing blocks (exact reference style)
    dash_props = dict(linestyle='--', lw=2.5, color='#FF9800', alpha=0.8)
    
    # Connect VGG to block 4
    ax.plot([11, 14], [2.8, 4.8], **dash_props)
    # Connect CGM to block 3  
    ax.plot([15.5, 16.5], [2.8, 4.8], **dash_props)
    # Connect Conv to block 2
    ax.plot([20.25, 19], [2.8, 4.8], **dash_props)
    
    # Remove all axes and spines for clean look
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    plt.tight_layout()
    plt.savefig('cgnet_architecture_exact_reference.png', dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none', pad_inches=0.1)
    plt.show()
    print("✅ CGNet architecture diagram created exactly matching the reference!")

if __name__ == "__main__":
    draw_cgnet_architecture()
