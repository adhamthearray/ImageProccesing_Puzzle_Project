import cv2
import numpy as np
import os

# ---------------- PATHS ----------------
# change these if your folders move
image_folder = r"E:\Semester 5\Image Processing\Project\Phase1\All sizes"
output_folder = r"E:\Semester 5\Image Processing\Project\Phase1\Outputpieces2"
os.makedirs(output_folder, exist_ok=True)

# ---------------- HELPER FUNCTIONS ----------------
def get_puzzle_size_from_folder(folder_name):
    folder_lower = folder_name.lower()
    if "8x8" in folder_lower:
        return 8, 8
    elif "4x4" in folder_lower:
        return 4, 4
    elif "2x2" in folder_lower:
        return 2, 2
    else:
        # default
        return 2, 2

def split_image(img, rows, cols):
    pieces = []
    h, w = img.shape[:2]
    h_step, w_step = h // rows, w // cols
    for i in range(rows):
        for j in range(cols):
            y1 = i * h_step
            y2 = h if i == rows - 1 else (i + 1) * h_step
            x1 = j * w_step
            x2 = w if j == cols - 1 else (i + 1) * w_step if False else (j + 1) * w_step
            piece = img[y1:y2, x1:x2]
            pieces.append(piece)
    return pieces

def analyze_piece(gray):
    mean_intensity = np.mean(gray)
    contrast = np.std(gray)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return mean_intensity, contrast, laplacian_var

def adaptive_parameters(mean_intensity, contrast, laplacian_var):
    # Bilateral parameters (same logic you had)
    if laplacian_var > 150:
        d, sigmaColor, sigmaSpace = 5, 30, 30
    elif laplacian_var > 70:
        d, sigmaColor, sigmaSpace = 7, 60, 60
    else:
        d, sigmaColor, sigmaSpace = 11, 90, 90

    # Adaptive threshold (kept for possible later use)
    if mean_intensity < 90:
        blockSize, C = 15, 3
    elif mean_intensity < 140:
        blockSize, C = 11, 5
    else:
        blockSize, C = 9, 2

    # Unsharp parameters
    if contrast < 30:
        alpha, beta = 1.8, -0.6
    elif contrast < 60:
        alpha, beta = 1.5, -0.5
    else:
        alpha, beta = 1.2, -0.3

    return d, sigmaColor, sigmaSpace, blockSize, C, alpha, beta

def unsharp_mask(image, alpha=1.5, beta=-0.5, sigma=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, alpha, blurred, beta, 0)

# ---------------- PREPROCESS ONE PIECE (ADAPTIVE) ----------------
def process_piece(piece):
    # 0. to gray
    gray = cv2.cvtColor(piece, cv2.COLOR_BGR2GRAY)

    # 1. stats + adaptive parameters
    mean_i, cont, lap_var = analyze_piece(gray)
    d, sc, ss, blockSize, C, alpha, beta = adaptive_parameters(mean_i, cont, lap_var)

    # 2. bilateral denoising
    filtered = cv2.bilateralFilter(gray, d, sc, ss)

    # 3. unsharp mask (adaptive strength)
    sharpened = unsharp_mask(filtered, alpha=alpha, beta=beta, sigma=1.0)

    # 4. CLAHE, clipLimit depends on contrast
    if cont < 30:
        clip = 4.0
    elif cont < 60:
        clip = 2.5
    else:
        clip = 1.8
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    enhanced = clahe.apply(sharpened)

    # 5. Canny thresholds based on stats
    median_val = np.median(enhanced)

    # contrast-dependent scaling
    if cont < 30:
        low_scale, high_scale = 0.6, 1.2
    elif cont < 60:
        low_scale, high_scale = 0.7, 1.4
    else:
        low_scale, high_scale = 0.9, 1.6

    lower = int(max(10,  low_scale  * median_val))
    upper = int(min(255, max(40, high_scale * median_val)))

    # blur before Canny
    smooth = cv2.GaussianBlur(enhanced, (3, 3), 0)
    edges = cv2.Canny(smooth, lower, upper, L2gradient=True)

    # 6. contour-based cleanup (remove tiny noisy contours)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(edges)

    # Perimeter threshold depends on sharpness (laplacian variance)
    if lap_var < 50:
        min_perim = 20
    elif lap_var < 150:
        min_perim = 30
    else:
        min_perim = 40

    for cnt in contours:
        if cv2.arcLength(cnt, True) > min_perim:
            cv2.drawContours(mask, [cnt], -1, 255, -1)

    edges_clean = mask

    return enhanced, edges_clean

# ---------------- MAIN PROCESSING LOOP ----------------
for root, dirs, files in os.walk(image_folder):
    for filename in files:
        if not filename.lower().endswith((".jpg", ".jpeg", ".png", ".bmp")):
            continue

        img_path = os.path.join(root, filename)
        img = cv2.imread(img_path)
        if img is None:
            print("Could not read:", img_path)
            continue

        rows, cols = get_puzzle_size_from_folder(os.path.basename(root))
        pieces = split_image(img, rows, cols)

        rel_path = os.path.relpath(root, image_folder)
        base_output = os.path.join(output_folder, rel_path, os.path.splitext(filename)[0])
        os.makedirs(base_output, exist_ok=True)

        print(f"Processing {filename} → {rows}x{cols} ({len(pieces)} pieces)")

        for idx, piece in enumerate(pieces):
            if piece.size == 0:
                continue

            gray_final, edge_final = process_piece(piece)

            cv2.imwrite(os.path.join(base_output, f"piece_{idx:02d}_gray.png"), gray_final)
            cv2.imwrite(os.path.join(base_output, f"piece_{idx:02d}_edge.png"), edge_final)
            cv2.imwrite(os.path.join(base_output, f"piece_{idx:02d}_original.png"), piece)

        print(f"   → Saved to {base_output}")

print("\nAll images processed successfully!")

# ---------------- VISUALIZATION (OPTIONAL) ----------------
def visualize_all_results():
    print("Building montages for visualization...")
    any_window = False

    for root, dirs, files in os.walk(output_folder):
        gray_files = sorted([f for f in os.listdir(root) if f.endswith("_gray.png")])
        if not gray_files:
            continue

        # grid size from number of pieces
        n = len(gray_files)
        if n == 4:
            rows, cols = 2, 2
        elif n == 16:
            rows, cols = 4, 4
        elif n == 64:
            rows, cols = 8, 8
        else:
            # not a full puzzle folder
            continue

        gray_pieces = []
        edge_pieces = []
        missing = False
        for i in range(rows * cols):
            idx = f"{i:02d}"
            g_path = os.path.join(root, f"piece_{idx}_gray.png")
            e_path = os.path.join(root, f"piece_{idx}_edge.png")
            if not (os.path.exists(g_path) and os.path.exists(e_path)):
                missing = True
                break
            gray_pieces.append(cv2.imread(g_path, cv2.IMREAD_GRAYSCALE))
            edge_pieces.append(cv2.imread(e_path, cv2.IMREAD_GRAYSCALE))
        if missing:
            continue

        # pad to same size
        max_h = max(p.shape[0] for p in gray_pieces)
        max_w = max(p.shape[1] for p in gray_pieces)

        def pad_to_size(img):
            pad_h = max_h - img.shape[0]
            pad_w = max_w - img.shape[1]
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            return cv2.copyMakeBorder(img, top, bottom, left, right,
                                      cv2.BORDER_CONSTANT, value=255)

        gray_pieces = [pad_to_size(p) for p in gray_pieces]
        edge_pieces = [pad_to_size(p) for p in edge_pieces]

        gray_montage = np.vstack([
            np.hstack(gray_pieces[i*cols:(i+1)*cols]) for i in range(rows)
        ])
        edge_montage = np.vstack([
            np.hstack(edge_pieces[i*cols:(i+1)*cols]) for i in range(rows)
        ])

        title = os.path.relpath(root, output_folder).replace("\\", "/")

        scale = 900 / max(gray_montage.shape)
        if scale < 1:
            gray_disp = cv2.resize(gray_montage, None, fx=scale, fy=scale)
            edge_disp = cv2.resize(edge_montage, None, fx=scale, fy=scale)
        else:
            gray_disp, edge_disp = gray_montage, edge_montage

        cv2.imshow(f"Grayscale - {title} [{rows}x{cols}]", gray_disp)
        cv2.imshow(f"Edges - {title} [{rows}x{cols}]", edge_disp)
        any_window = True

    if any_window:
        print("Visualization ready! Close any window to exit.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No complete puzzle found for visualization.")

# uncomment if you want to auto-run the visualization:
# visualize_all_results()
