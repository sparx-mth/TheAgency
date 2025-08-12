import os
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from scipy import ndimage
from skimage.measure import label, regionprops
from skimage.morphology import binary_opening, binary_closing
import argparse

class BuildingMapAnalyzer:
    def __init__(self):
        # Color thresholds for different map elements (BGR format for OpenCV)
        self.color_thresholds = {
            'unknown': ([0, 0, 0], [50, 50, 50]),        # Black - wider range
            'free': ([60, 60, 60], [130, 130, 130]),     # Dark gray - wider range
            'wall': ([120, 120, 120], [170, 170, 170]),  # Medium gray - wider range
            'door': ([170, 170, 170], [230, 230, 230]),  # Light gray - wider range
            'static': ([230, 230, 230], [255, 255, 255]) # White
        }
        
    def analyze_colors(self, image):
        """Debug function to analyze actual colors in image"""
        unique_colors = {}
        h, w = image.shape[:2]
        
        # Sample colors from image
        for y in range(0, h, 10):
            for x in range(0, w, 10):
                color = tuple(image[y, x])
                if color in unique_colors:
                    unique_colors[color] += 1
                else:
                    unique_colors[color] = 1
        
        # Sort by frequency
        sorted_colors = sorted(unique_colors.items(), key=lambda x: x[1], reverse=True)
        print("Top colors in image:")
        for i, (color, count) in enumerate(sorted_colors[:10]):
            print(f"  {i+1}. RGB{color} - {count} pixels")
        
        return sorted_colors
    
    def load_image(self, image_path):
        """Load and return image in BGR format"""
        return cv2.imread(image_path)
    
    def create_mask(self, image, color_type):
        """Create binary mask for specific color type"""
        lower, upper = self.color_thresholds[color_type]
        lower = np.array(lower, dtype=np.uint8)
        upper = np.array(upper, dtype=np.uint8)
        return cv2.inRange(image, lower, upper)
    
    def detect_rooms(self, image):
        """Detect rooms as enclosed areas surrounded by walls (4 sides)"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Find free space - pixels that are not black (unknown) and not white (static)
        #free_mask = ((gray > 50) & (gray < 200)).astype(np.uint8) * 255
        free_mask = (self.create_mask(image, 'free') > 0).astype(np.uint8) * 255

        print(f"Free mask pixels found: {np.sum(free_mask == 255)}")
        
        # Use flood fill from edges to remove areas connected to borders
        # This leaves only truly enclosed rooms
        h, w = gray.shape
        mask_copy = free_mask.copy()
        
        # Flood fill from all border pixels
        for i in range(h):
            # Left and right edges
            if mask_copy[i, 0] == 255:
                cv2.floodFill(mask_copy, None, (0, i), 0)
            if mask_copy[i, w-1] == 255:
                cv2.floodFill(mask_copy, None, (w-1, i), 0)
        
        for j in range(w):
            # Top and bottom edges  
            if mask_copy[0, j] == 255:
                cv2.floodFill(mask_copy, None, (j, 0), 0)
            if mask_copy[h-1, j] == 255:
                cv2.floodFill(mask_copy, None, (j, h-1), 0)
        
        # What's left are enclosed areas (rooms)
        enclosed_mask = mask_copy
        
        # Clean up small artifacts
        kernel = np.ones((3,3), np.uint8)
        enclosed_mask = cv2.morphologyEx(enclosed_mask, cv2.MORPH_OPEN, kernel)
        
        # Find connected components
        labeled = label(enclosed_mask)
        regions = regionprops(labeled)
        
        print(f"Found {len(regions)} potential enclosed rooms")
        
        rooms = []
        for i, region in enumerate(regions):
            if region.area > 100:  # Minimum room size
                minr, minc, maxr, maxc = region.bbox
                print(f"Room {i+1}: area={region.area}, bbox=({minc},{minr},{maxc},{maxr})")
                rooms.append({
                    'id': i + 1,
                    'area_px': int(region.area),
                    'bbox': [int(minc), int(minr), int(maxc), int(maxr)],
                    'centroid': [int(region.centroid[1]), int(region.centroid[0])]
                })
        
        return rooms
    
    def detect_corridors(self, image):
        """Skip corridor detection as requested"""
        print("Skipping corridor detection")
        return []

    def detect_doorways(self, image):
        """Detect doorways as light-gray pixels connecting free spaces"""
        door_mask = self.create_mask(image, 'door')
        
        print(f"Door pixels found: {np.sum(door_mask == 255)}")
        
        if np.sum(door_mask == 255) == 0:
            # Try alternative door detection - find medium-light gray areas
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            door_mask = ((gray > 150) & (gray < 220)).astype(np.uint8) * 255
            print(f"Alternative door pixels found: {np.sum(door_mask == 255)}")
        
        # Find door pixels
        door_coords = np.where(door_mask == 255)
        doorways = []
        
        # Group nearby door pixels
        if len(door_coords[0]) > 0:
            labeled = label(door_mask)
            regions = regionprops(labeled)
            
            for region in regions:
                if region.area > 1:  # At least 1 pixel
                    centroid_y, centroid_x = region.centroid
                    doorways.append({
                        'x': int(centroid_x),
                        'y': int(centroid_y)
                    })
        
        print(f"Found {len(doorways)} doorways")
        return doorways

    def detect_doorways_2(self, image):
        """Detect doorways as lighter pixels that create passages between wall pixels"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Find wall pixels (medium gray)
        wall_mask = ((gray >= 120) & (gray <= 170)).astype(np.uint8) * 255
        
        # Find potential door pixels (lighter than walls but not white)
        door_candidates = ((gray > 150) & (gray < 240)).astype(np.uint8) * 255
        
        print(f"Wall pixels: {np.sum(wall_mask == 255)}")
        print(f"Door candidate pixels: {np.sum(door_candidates == 255)}")
        
        doorways = []
        
        # Check each door candidate pixel
        door_coords = np.where(door_candidates == 255)
        
        for i in range(len(door_coords[0])):
            y, x = door_coords[0][i], door_coords[1][i]
            
            # Check if this pixel connects wall segments
            # Look in 4 directions around the pixel
            neighbors = []
            if y > 0: neighbors.append(wall_mask[y-1, x])
            if y < h-1: neighbors.append(wall_mask[y+1, x])  
            if x > 0: neighbors.append(wall_mask[y, x-1])
            if x < w-1: neighbors.append(wall_mask[y, x+1])
            
            # If surrounded by walls on at least 2 sides, likely a door
            wall_neighbors = sum(n == 255 for n in neighbors)
            
            if wall_neighbors >= 2:
                doorways.append({
                    'x': int(x),
                    'y': int(y)
                })
        
        # Remove duplicate nearby doorways
        filtered_doorways = []
        for door in doorways:
            is_duplicate = False
            for existing in filtered_doorways:
                if abs(door['x'] - existing['x']) <= 2 and abs(door['y'] - existing['y']) <= 2:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_doorways.append(door)
        
        print(f"Found {len(filtered_doorways)} doorways")
        return filtered_doorways
    
    def detect_obstacles(self, image):
        """Detect obstacles as bright pixels completely surrounded by dark pixels"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Find very bright pixels (potential obstacles)
        bright_mask = (gray > 240).astype(np.uint8) * 255
        
        print(f"Bright pixels found: {np.sum(bright_mask == 255)}")
        
        obstacles = []
        bright_coords = np.where(bright_mask == 255)
        
        for i in range(len(bright_coords[0])):
            y, x = bright_coords[0][i], bright_coords[1][i]
            
            # Check if completely surrounded by dark pixels
            surrounded_by_dark = True
            
            # Check 8-neighborhood
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0:
                        continue
                    
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w:
                        # If neighbor is not dark (>80), not surrounded
                        if gray[ny, nx] > 80:
                            surrounded_by_dark = False
                            break
                if not surrounded_by_dark:
                    break
            
            if surrounded_by_dark:
                obstacles.append({
                    'x': int(x),
                    'y': int(y),
                    'area': 1
                })
        
        # Remove duplicate nearby obstacles
        filtered_obstacles = []
        for obs in obstacles:
            is_duplicate = False
            for existing in filtered_obstacles:
                if abs(obs['x'] - existing['x']) <= 1 and abs(obs['y'] - existing['y']) <= 1:
                    is_duplicate = True
                    break
            if not is_duplicate:
                filtered_obstacles.append(obs)
        
        print(f"Found {len(filtered_obstacles)} obstacles")
        return filtered_obstacles
    
    def analyze_map(self, image_path):
        """Main analysis function"""
        image = self.load_image(image_path)
        if image is None:
            return None
        
        print(f"\n=== Analyzing {image_path} ===")
        print(f"Image shape: {image.shape}")
        
        # Debug: analyze actual colors in the image
        self.analyze_colors(image)
        
        # Create debug masks to see what's being detected
        print("\nCreating masks...")
        for color_type in ['unknown', 'free', 'wall', 'door', 'static']:
            mask = self.create_mask(image, color_type)
            pixels = np.sum(mask == 255)
            print(f"{color_type}: {pixels} pixels")
        
        results = {
            'rooms': self.detect_rooms(image),
            'corridors': self.detect_corridors(image),
            'doorways': self.detect_doorways(image),
            'obstacles': self.detect_obstacles(image)
        }
        
        print(f"\nResults summary:")
        print(f"Rooms: {len(results['rooms'])}")
        print(f"Corridors: {len(results['corridors'])}")
        print(f"Doorways: {len(results['doorways'])}")  
        print(f"Obstacles: {len(results['obstacles'])}")
        
        return results
    
    def create_annotated_image(self, image_path, analysis_results, output_path):
        """Create annotated image with bounding boxes and labels"""
        # Load image using PIL for better drawing capabilities
        img = Image.open(image_path)
        draw = ImageDraw.Draw(img)
        
        # Try to use a better font, fall back to default if not available
        try:
            font = ImageFont.truetype("arial.ttf", 12)
        except:
            font = ImageFont.load_default()
        
        # Colors for different elements
        colors = {
            'rooms': 'blue',
            'corridors': 'orange', 
            'doorways': 'magenta',
            'obstacles': 'red'
        }
        
        # Draw rooms
        for i, room in enumerate(analysis_results['rooms']):
            bbox = room['bbox']
            draw.rectangle(bbox, outline=colors['rooms'], width=2)
            draw.text((bbox[0], bbox[1]-15), f"Room {room['id']}", 
                     fill=colors['rooms'], font=font)
        
        # Draw corridors (skip since we removed corridor detection)
        # No corridors to draw
        
        # Draw doorways
        for i, doorway in enumerate(analysis_results['doorways']):
            x, y = doorway['x'], doorway['y']
            draw.rectangle([x-3, y-3, x+3, y+3], outline=colors['doorways'], width=2)
            draw.text((x+5, y-10), f"D{i+1}", fill=colors['doorways'], font=font)
        
        # Draw obstacles
        for i, obstacle in enumerate(analysis_results['obstacles']):
            x, y = obstacle['x'], obstacle['y']
            draw.rectangle([x-4, y-4, x+4, y+4], outline=colors['obstacles'], width=2)
            draw.text((x+6, y-10), f"O{i+1}", fill=colors['obstacles'], font=font)
        
        # Save annotated image
        img.save(output_path)
        print(f"Annotated image saved: {output_path}")
    
    def process_directory(self, input_dir, output_dir):
        """Process all images in directory and create JSON dataset"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        annotated_dir = os.path.join(output_dir, 'annotated')
        if not os.path.exists(annotated_dir):
            os.makedirs(annotated_dir)
        
        dataset = []
        
        # Supported image formats
        image_extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
        
        for filename in os.listdir(input_dir):
            if any(filename.lower().endswith(ext) for ext in image_extensions):
                print(f"Processing {filename}...")
                
                # Create paths
                image_path = os.path.join(input_dir, filename)
                name_without_ext = os.path.splitext(filename)[0]
                
                # Analyze the map
                results = self.analyze_map(image_path)
                if results is None:
                    print(f"Failed to process {filename}")
                    continue
                
                # Create annotated image
                annotated_path = os.path.join(annotated_dir, f"{name_without_ext}_annotated.png")
                self.create_annotated_image(image_path, results, annotated_path)
                
                # Create dataset entry
                dataset_entry = {
                    "id": name_without_ext,
                    "image": f"data/maps/{filename}",
                    "conversations": [
                        {
                            "from": "human",
                            "value": "You are a building-map analyst.\nLegend: black=unknown, dark-gray=free, medium-gray=wall, light-gray=door, white=static.\nTasks: (1) Detect rooms (free regions bounded by walls/unknown), (2) Detect corridors (narrow free passages) and give approx length in pixels, (3) Detect doorways = light-gray pixels connecting free space on both sides of a wall line, (4) Detect obstacles = static pixels inside free space.\nOutput JSON only with keys: rooms, corridors, doorways, obstacles."
                        },
                        {
                            "from": "gpt",
                            "value": json.dumps(results, separators=(',', ':'))
                        }
                    ]
                }
                
                dataset.append(dataset_entry)
                print(f"✓ Completed {filename}")
        
        # Save dataset JSON
        dataset_path = os.path.join(output_dir, 'building_maps_dataset.json')
        with open(dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)
        
        print(f"\n✓ Dataset saved: {dataset_path}")
        print(f"✓ Processed {len(dataset)} images")
        print(f"✓ Annotated images saved in: {annotated_dir}")

def main():
    parser = argparse.ArgumentParser(description='Analyze building maps and create training dataset')
    parser.add_argument('input_dir', help='Directory containing building map images')
    parser.add_argument('output_dir', help='Directory to save results')
    parser.add_argument('--single', help='Process single image file', default=None)
    
    args = parser.parse_args()
    
    analyzer = BuildingMapAnalyzer()
    
    if args.single:
        # Process single image
        results = analyzer.analyze_map(args.single)
        if results:
            print("Analysis Results:")
            print(json.dumps(results, indent=2))
            
            # Create annotated image
            name = os.path.splitext(os.path.basename(args.single))[0]
            output_path = f"{name}_annotated.png"
            analyzer.create_annotated_image(args.single, results, output_path)
    else:
        # Process directory
        analyzer.process_directory(args.input_dir, args.output_dir)

if __name__ == "__main__":
    main()