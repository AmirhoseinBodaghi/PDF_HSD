import cv2
import numpy as np
import fitz  # PyMuPDF
import pytesseract
import matplotlib.pyplot as plt
from skimage.feature import hog
import streamlit as st
from PIL import Image
from io import BytesIO

def extract_images_from_pdf(pdf_path):
    """Extracts images from a PDF file."""
    pdf_document = fitz.open(pdf_path)
    images = []
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        pix = page.get_pixmap()
        img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, pix.n)
        # Make the image writable
        img = np.copy(img)
        images.append(img)
    return images

def segment_subregions(roi):
    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    # Improve contrast using histogram equalization
    gray = cv2.equalizeHist(gray)

    # Apply adaptive thresholding to handle varying illumination
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Edge detection with refined parameters
    edges = cv2.Canny(binary, 300, 500)  # Lower thresholds for more sensitivity

    # Find contours with tree hierarchy for nested contours
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    subregions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h

        # Filter out noise with lower area thresholds
        if area > 1000:  # Lower the threshold to capture smaller details
            subregions.append({'x': x, 'y': y, 'width': w, 'height': h})

    return subregions

def merge_overlapping_regions(regions):
    """
    Merges overlapping regions by grouping connected regions and creating a single bounding box for each group.
    """
    def overlap(region1, region2):
        """Check if two regions overlap."""
        x1, y1, w1, h1 = region1['x'], region1['y'], region1['width'], region1['height']
        x2, y2, w2, h2 = region2['x'], region2['y'], region2['width'], region2['height']
        return (x1 < x2 + w2 and x1 + w1 > x2 and
                y1 < y2 + h2 and y1 + h1 > y2)

    # Group overlapping regions
    groups = []
    for region in regions:
        matched = False
        for group in groups:
            if any(overlap(region, other) for other in group):
                group.append(region)
                matched = True
                break
        if not matched:
            groups.append([region])

    # Merge each group into a single bounding box
    merged_regions = []
    for group in groups:
        if len(group) == 1:
            merged_regions.append(group[0])
            continue

        # Compute a single bounding box for the group
        x_min = min(r['x'] for r in group)
        y_min = min(r['y'] for r in group)
        x_max = max(r['x'] + r['width'] for r in group)
        y_max = max(r['y'] + r['height'] for r in group)

        merged_regions.append({
            'x': x_min,
            'y': y_min,
            'width': x_max - x_min,
            'height': y_max - y_min
        })

    return merged_regions


def extract_text_features(roi):
    """Extracts features to distinguish between handwritten and typed text."""
    # 1. OCR Bounding Boxes
    boxes = pytesseract.image_to_boxes(roi)
    heights = []
    widths = []
    baselines = []
    aspect_ratios = []
    spacings = []

    prev_x2 = None  # Previous box's right boundary (x2)

    for box in boxes.splitlines():
        b = box.split()
        x1, y1, x2, y2 = int(b[1]), int(b[2]), int(b[3]), int(b[4])
        width = x2 - x1
        height = y2 - y1
        baseline = y1
        if height != 0:
            aspect_ratio = width / height
        else:
            aspect_ratio = 0

        # Collect height, width, baseline, and aspect ratio
        heights.append(height)
        widths.append(width)
        baselines.append(baseline)
        aspect_ratios.append(aspect_ratio)

        # Calculate spacing (if there is a previous character)
        if prev_x2 is not None:
            spacings.append(x1 - prev_x2)
        prev_x2 = x2

    # 2. Calculate Variance and Consistency Features
    height_variance = np.var(heights) if heights else 0
    width_variance = np.var(widths) if widths else 0
    baseline_variance = np.var(baselines) if baselines else 0
    aspect_ratio_variance = np.var(aspect_ratios) if aspect_ratios else 0
    spacing_variance = np.var(spacings) if spacings else 0
    avg_spacing = np.mean(spacings) if spacings else 0

    # 3. Edge Features
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (roi.shape[0] * roi.shape[1])

    # Return features
    return {
        "height_variance": height_variance,
        "width_variance": width_variance,
        "baseline_variance": baseline_variance,
        "aspect_ratio_variance": aspect_ratio_variance,
        "spacing_variance": spacing_variance,
        "avg_spacing": avg_spacing,
        "edge_density": edge_density,
    }


def region_classification (edge_density, avg_confidence, text_length):    
    # Default classification
    classification = "typed"

    # Classification rules
    if edge_density > 0.169:
        classification = "typed"
    elif 0 <= avg_confidence <= 10:
        classification = "handwritten"
    elif 0 < text_length <= 4:
        classification = "handwritten"
    elif 0 <= avg_confidence <= 20 and text_length <= 11 and edge_density <= 0.08:
        classification = "handwritten"
    elif avg_confidence == -1.0 and text_length == 0:
        classification = "typed"

    return classification

def subregion_classification(avg_confidence, text_length):
    classification = "typed"
    
    if 0 <= avg_confidence <= 10 and text_length <= 5:
        classification = "handwritten"
    
    else:
        classification = "typed"
    
    return classification

def extract_golden_features (roi):
    roi_features = extract_text_features(roi)
    edge_density = roi_features['edge_density']

    # 2- text length
    config = '--psm 6'  # Adjust PSM mode for better OCR
    text = pytesseract.image_to_string(roi, config=config)
    text_length = len(text.strip())

    # 3- OCR confidence details
    details = pytesseract.image_to_data(roi, config=config, output_type=pytesseract.Output.DICT)
    confidence_scores = details['conf']
    avg_confidence = (
        sum([int(c) for c in confidence_scores if c != '-1']) / len(confidence_scores)
        if confidence_scores else 0
    )
    return edge_density, text_length, avg_confidence


def detect_signature(image): 
    """Detects signature regions in the image using contour detection and OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to handle varying illumination
    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Edge detection with refined thresholds
    edges = cv2.Canny(binary, 30, 100)

    # Apply morphological dilation to close gaps in contours
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    signature_regions = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)

        # Filter out small contours (noise, irrelevant text)
        if area < 500:  # Increase minimum area threshold to filter small contours
            continue

        # Add bounding box details
        signature_regions.append({'x': x, 'y': y, 'width': w, 'height': h})

    # Merge overlapping signature regions
    merged_regions = merge_overlapping_regions(signature_regions)

##    
##    print("\nMerged Regions:")
##    for idx, region in enumerate(merged_regions):
##        print(f"Region {idx + 1}: (x={region['x']}, y={region['y']}, "
##              f"width={region['width']}, height={region['height']})")
##    

    # Analyze merged regions
    analyzed_regions = []
    for idx, region in enumerate(merged_regions):
        x, y, w, h = region['x'], region['y'], region['width'], region['height']
        roi = image[y:y + h, x:x + w]

        # Extract three golden features:
        edge_density, text_length, avg_confidence = extract_golden_features (roi)
      
        classification = region_classification (edge_density, avg_confidence, text_length)
        # Draw rectangle based on classification

        # if you want to see non handwritten regions, then uncomment the below line
        # region_color = (0, 255, 0) if classification == "handwritten" else (255, 0, 0)  # Green for handwritten, blue for typed
        # cv2.rectangle(image, (x, y), (x + w, y + h), region_color, 2)

        # if you want to see non handwritten regions, then comment the below if
        if classification == "handwritten":
            # Draw rectangle for handwritten regions only
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for handwritten
            # Add a label for handwritten regions only
            # label = f"ID: {idx + 1}"
            # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Add a label with the region ID
        # label = f"ID: {idx + 1}"
        # cv2.putText(image, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
        ## Print region details
        # print(f"Region {idx + 1}:")
        # print(f"  - Location: (x={x}, y={y}, width={w}, height={h})")
        # print(f"  - Area: {w * h}")
        # print(f"  - Average OCR Confidence: {avg_confidence:.2f}")
        # print(f"  - Text Length: {text_length}")
        # print(f"  - Edge Density: {edge_density}")   
        # print(f"  - Classification: {classification}")
        # print(f"  - *******************:")
        

        # Sub Region Analysis
        subregions = segment_subregions(roi)
        # Merge overlapping subregions
        merged_subregions = merge_overlapping_regions(subregions)
        results = []

        for sub_idx, subregion in enumerate(merged_subregions, start=1):
            # Subregion coordinates relative to the ROI
            sub_x, sub_y, sub_w, sub_h = subregion['x'], subregion['y'], subregion['width'], subregion['height']
            
            # Convert subregion coordinates to global coordinates
            global_x = region['x'] + sub_x
            global_y = region['y'] + sub_y

            # Extract sub-region ROI
            sub_roi = roi[sub_y:sub_y + sub_h, sub_x:sub_x + sub_w]
            
            # Extract features and classify
            edge_density, text_length, avg_confidence = extract_golden_features(sub_roi)
            classification = subregion_classification(avg_confidence, text_length)
            results.append({
                'x': global_x,
                'y': global_y,
                'width': sub_w,
                'height': sub_h,
                'classification': classification
            })
            
            # Draw rectangle on the global image
            # region_color = (0, 255, 0) if classification == "handwritten" else (255, 0, 0)  # Green for handwritten, blue for typed
            # cv2.rectangle(image, (global_x, global_y), (global_x + sub_w, global_y + sub_h), region_color, 2)

            if classification == "handwritten":
                # Draw rectangle for handwritten subregions only
                cv2.rectangle(image, (global_x, global_y), (global_x + sub_w, global_y + sub_h), (0, 255, 0), 2)
                # Add a label for handwritten subregions only
                # sub_label = f"Sub-ID: {idx + 1}-{sub_idx}"
                # cv2.putText(image, sub_label, (global_x, global_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
            
            
            # Add a label with the subregion ID
            # sub_label = f"Sub-ID: {idx + 1}-{sub_idx}"
            # cv2.putText(image, sub_label, (global_x, global_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

##            
##            # Print subregion details
##            print(f"  Subregion {sub_idx} of Region {idx + 1}:")
##            print(f"    - Global Location: (x={global_x}, y={global_y}, width={sub_w}, height={sub_h})")
##            print(f"    - Area: {sub_w * sub_h}")
##            print(f"    - Average OCR Confidence: {avg_confidence:.2f}")
##            print(f"    - Text Length: {text_length}")
##            print(f"    - Edge Density: {edge_density}")
##            print(f"    - Classification: {classification}")
##            print(f"    - -------------------------")
##            

    # Visualization: Draw the detected signature regions
    output_image = image.copy()
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
    plt.title("Detected Handwritten Signature Regions with Subregion Labels")
    plt.axis("off")
    plt.show()

    return image


##
##def process_pdf(pdf_path):
##    images = extract_images_from_pdf(pdf_path)
##    all_results = []
##    
##    for page_num, image in enumerate(images):
##        print(f"Processing page {page_num + 1}...")
##        
##        # Detect signatures
##        signature_marked_image = detect_signature(image)
##        
##        
##    return all_results
##
##
##
### Example usage
##pdf_path = 'FP1.pdf'  # Path to the input PDF file
##results = process_pdf(pdf_path)
##
### Display results
##for result in results:
##    print(f"\nPage {result['page']} Results:")
##    print(f"Signatures: {result['signatures']}")
##

# Streamlit App
st.title("Handwritten Signature Detection in PDFs")
st.write("Please upload a PDF document to detect handwritten signatures.")

# File upload
uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    # Save the uploaded file to a temporary location
    pdf_bytes = uploaded_file.read()
    pdf_path = "uploaded_document.pdf"
    with open(pdf_path, "wb") as f:
        f.write(pdf_bytes)

    # Process the uploaded PDF
    st.write("Processing the PDF...")
    images = extract_images_from_pdf(pdf_path)

    for page_num, image in enumerate(images):
        st.write(f"### Page {page_num + 1}")

        # Detect signatures
        image = detect_signature(image)

        # Display the marked image
        st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), caption=f"Page {page_num + 1} Processed", use_container_width=True)

        
##        # Display signature details
##        if signature_regions:
##            st.write("**Detected Signatures:**")
##            for region in signature_regions:
##                st.write(f"- Region ID: {region['id']}")
##                st.write(f"  - Location: (x={region['x']}, y={region['y']}, width={region['width']}, height={region['height']})")
##                st.write(f"  - Classification: {region['classification']}")
##                st.write(f"  - Area: {region['area']}")
##                st.write(f"  - OCR Confidence: {region['avg_confidence']:.2f}")
##                st.write(f"  - Aspect Ratio: {region['aspect_ratio']:.2f}")
##        else:
##            st.write("No signatures detected on this page.")
##        

    st.success("Processing completed.")
