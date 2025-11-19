"""
Create placeholder images for the server
"""
import matplotlib.pyplot as plt
import numpy as np
import os

# Ensure output directory exists
output_dir = "src/pendantprop/server/static/placehold_images"
os.makedirs(output_dir, exist_ok=True)

# Create dynamic surface tension placeholder
fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, 'No Dynamic Surface Tension Data\nStart a measurement to see results', 
        ha='center', va='center', fontsize=14, color='gray')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{output_dir}/dynamic_surface_tension_plot_1.png", dpi=100, bbox_inches='tight')
plt.close()

# Create results plot placeholder
fig, ax = plt.subplots(figsize=(8, 6))
ax.text(0.5, 0.5, 'No Results Data\nComplete measurements to see equilibrium surface tension', 
        ha='center', va='center', fontsize=14, color='gray')
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis('off')
plt.tight_layout()
plt.savefig(f"{output_dir}/results_plot.png", dpi=100, bbox_inches='tight')
plt.close()

print("Placeholder images created successfully!")
print(f"Location: {output_dir}")
