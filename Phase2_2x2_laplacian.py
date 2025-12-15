import os
import cv2
import numpy as np
from itertools import permutations

# ==========================================================
# STEP 1 — COLLECT ALL *_edge.png
# ==========================================================
def collect_edge_images(root_folder):
    edge_pieces = []
    for root, dirs, files in os.walk(root_folder):
        for f in files:
            if f.lower().endswith("_edge.png"):
                full_path = os.path.join(root, f)
                img = cv2.imread(full_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    print("Could not load:", full_path)
                    continue
                _, img_bin = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
                edge_pieces.append({"path": full_path, "image": img_bin})
    return edge_pieces

# ==========================================================
# STEP 2 — EXTRACT SIDES
# ==========================================================
def extract_sides(edge_img, border=5):
    h, w = edge_img.shape
    top = edge_img[0:border, :]
    bottom = edge_img[-border:, :]
    left = edge_img[:, 0:border]
    right = edge_img[:, -border:]
    return top, right, bottom, left

def extract_sides_color(img, border=5):
    h, w = img.shape[:2]
    top = img[0:border, :, :]
    bottom = img[-border:, :, :]
    left = img[:, 0:border, :]
    right = img[:, -border:, :]
    return top, right, bottom, left

# ==========================================================
# STEP 2.5 — LAPLACIAN EDGE ENHANCEMENT (COLOR PIECES)
# ==========================================================
def laplacian_enhance_bgr(img_bgr, alpha=0.8, ksize=3):
    """
    Enhance edges while keeping colors:
      enhanced = img + alpha * Laplacian(gray)

    alpha: strength (0.4..1.2 typical)
    ksize: Laplacian kernel size (1 or 3 usually)
    """
    # Mild denoise so Laplacian doesn't amplify noise too much
    smooth = cv2.bilateralFilter(img_bgr, d=7, sigmaColor=50, sigmaSpace=50)

    gray = cv2.cvtColor(smooth, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_16S, ksize=ksize)
    lap = cv2.convertScaleAbs(lap)

    lap3 = cv2.merge([lap, lap, lap])
    enhanced = cv2.addWeighted(smooth, 1.0, lap3, alpha, 0)
    return enhanced

# ==========================================================
# STEP 3 — EDGE DISTANCE FUNCTIONS
# ==========================================================
def edge_distance_pixels(edge1, edge2):
    diff_normal = np.sum(np.abs(edge1.astype(int) - edge2.astype(int)))
    diff_flipped = np.sum(np.abs(edge1.astype(int) - edge2[::-1, :].astype(int)))
    return min(diff_normal, diff_flipped)

def edge_distance_color_hist(edge1, edge2):
    # Convert to HSV
    e1 = cv2.cvtColor(edge1, cv2.COLOR_BGR2HSV)
    e2 = cv2.cvtColor(edge2, cv2.COLOR_BGR2HSV)
    e2_flip = cv2.flip(e2, 0)

    # Mask background by ignoring most frequent color
    def mask_background(edge):
        pixels = edge.reshape(-1, 3)
        counts = {}
        for px in map(tuple, pixels):
            counts[px] = counts.get(px, 0) + 1
        bg_color = max(counts, key=counts.get)
        mask = np.any(edge != bg_color, axis=2).astype(np.uint8)
        return mask

    mask1 = mask_background(e1)
    mask2 = mask_background(e2)
    mask2f = mask_background(e2_flip)

    hist1 = cv2.calcHist([e1], [0, 1, 2], mask1, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist2 = cv2.calcHist([e2], [0, 1, 2], mask2, [8, 8, 8], [0, 180, 0, 256, 0, 256])
    hist2f = cv2.calcHist([e2_flip], [0, 1, 2], mask2f, [8, 8, 8], [0, 180, 0, 256, 0, 256])

    cv2.normalize(hist1, hist1)
    cv2.normalize(hist2, hist2)
    cv2.normalize(hist2f, hist2f)

    sim1 = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
    sim2 = cv2.compareHist(hist1, hist2f, cv2.HISTCMP_CORREL)

    best_sim = max(sim1, sim2)
    return 1 - best_sim  # 0 = perfect match, larger = worse

# ==========================================================
# STEP 4 — LOAD PIECES (ORIGINAL CODE + LAPLACIAN ENHANCE)
# ==========================================================
def load_pieces(folder, border=5, alpha=0.8, ksize=3):
    edges = collect_edge_images(folder)
    pieces = []

    for idx, item in enumerate(edges):
        original_path = item["path"].replace("_edge.png", "_original.png")
        original = cv2.imread(original_path)
        if original is None:
            print("Missing original:", original_path)
            continue

        # grayscale edge strips (from *_edge.png)
        top, right, bottom, left = extract_sides(item["image"], border=border)

        # NEW: enhance edges on the colored original (but keep full colored image)
        enhanced = laplacian_enhance_bgr(original, alpha=alpha, ksize=ksize)

        # color edge strips (from enhanced image)
        ot, orr, ob, ol = extract_sides_color(enhanced, border=border)

        pieces.append({
            "id": idx,
            "edge_path": item["path"],
            "original_path": original_path,
            "top": top, "right": right, "bottom": bottom, "left": left,
            "ot": ot, "or": orr, "ob": ob, "ol": ol,
            # optional: keep enhanced for debug/visualization if you want later
            "enhanced": enhanced,
        })
    return pieces

# ==========================================================
# STEP 5 — TOTAL ERROR (Hybrid for 2x2)
# ==========================================================
def total_error(arr):
    TL, TR, BL, BR = arr
    ERR = 0

    # Masked HSV histogram similarity on enhanced color strips
    ERR += edge_distance_color_hist(TL["or"], TR["ol"])
    ERR += edge_distance_color_hist(TL["ob"], BL["ot"])
    ERR += edge_distance_color_hist(TR["ob"], BR["ot"])
    ERR += edge_distance_color_hist(BL["or"], BR["ol"])

    return ERR

# ==========================================================
# STEP 5b — COMPUTE "BAD SIDES" & CHOOSE FIRST PIECE
# ==========================================================
def compute_best_side_costs(pieces):
    N = len(pieces)
    INF = 1e9
    best_top    = np.full(N, INF, dtype=np.float32)
    best_right  = np.full(N, INF, dtype=np.float32)
    best_bottom = np.full(N, INF, dtype=np.float32)
    best_left   = np.full(N, INF, dtype=np.float32)

    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            cost_down = edge_distance_color_hist(pieces[i]["ob"], pieces[j]["ot"])
            if cost_down < best_bottom[i]:
                best_bottom[i] = cost_down

            cost_up = edge_distance_color_hist(pieces[i]["ot"], pieces[j]["ob"])
            if cost_up < best_top[i]:
                best_top[i] = cost_up

            cost_right = edge_distance_color_hist(pieces[i]["or"], pieces[j]["ol"])
            if cost_right < best_right[i]:
                best_right[i] = cost_right

            cost_left = edge_distance_color_hist(pieces[i]["ol"], pieces[j]["or"])
            if cost_left < best_left[i]:
                best_left[i] = cost_left

    return best_top, best_right, best_bottom, best_left

def choose_first_piece_top_left(pieces):
    best_top, best_right, best_bottom, best_left = compute_best_side_costs(pieces)

    all_vals = np.concatenate([best_top, best_right, best_bottom, best_left])
    thr = np.percentile(all_vals, 75)

    candidates = []
    for i in range(len(pieces)):
        bad_top  = best_top[i]  > thr
        bad_left = best_left[i] > thr
        if bad_top and bad_left:
            score = best_top[i] + best_left[i]
            candidates.append((score, i))

    if candidates:
        candidates.sort(reverse=True)
        start_idx = candidates[0][1]
    else:
        scores = [(best_top[i] + best_left[i], i) for i in range(len(pieces))]
        scores.sort(reverse=True)
        start_idx = scores[0][1]

    return start_idx, (best_top, best_right, best_bottom, best_left), thr

# ==========================================================
# STEP 6 — BRUTE FORCE SOLVER (2x2)
# ==========================================================
def optimize_2x2(pieces):
    best_arrangement = pieces[:4]
    best_error = total_error(best_arrangement)

    for perm in permutations(pieces[:4]):
        err = total_error(perm)
        if err < best_error:
            best_error = err
            best_arrangement = perm

    return best_arrangement, best_error

# ==========================================================
# STEP 7 — VISUALIZATION
# ==========================================================
def visualize_2x2_original(TL, TR, BL, BR, border=5, show=True):
    # IMPORTANT: use ORIGINAL images so output looks like your second example
    imgs = [cv2.imread(p["original_path"]) for p in [TL, TR, BL, BR]]

    h = max(img.shape[0] for img in imgs)
    w = max(img.shape[1] for img in imgs)
    imgs_resized = [cv2.resize(img, (w, h)) for img in imgs]

    black_v = np.zeros((h, border, 3), dtype=np.uint8)
    black_h = np.zeros((border, w*2 + border, 3), dtype=np.uint8)

    top_row = np.hstack([imgs_resized[0], black_v, imgs_resized[1]])
    bottom_row = np.hstack([imgs_resized[2], black_v, imgs_resized[3]])
    full = np.vstack([top_row, black_h, bottom_row])

    if show:
        cv2.imshow("2x2 Puzzle", full)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return full

# ==========================================================
# STEP 8 — MAIN
# ==========================================================
if __name__ == "__main__":
    base_folder = r"D:\Semester 5\Image processing\Project\Image_Milestone2\output_to_collect"
    output_folder = r"D:\Semester 5\Image processing\Project\Image_Milestone2\output_to_collect\assembled_laplacian"
    os.makedirs(output_folder, exist_ok=True)

    # You can tune these 3 only:
    BORDER = 5        # try 3, 4, 5, 6
    ALPHA  = 0.8      # try 0.4, 0.8, 1.2
    KSIZE  = 3        # 1 or 3

    for i in range(110):
        folder = os.path.join(base_folder, str(i))
        print("\nProcessing folder:", folder)

        pieces = load_pieces(folder, border=BORDER, alpha=ALPHA, ksize=KSIZE)
        print("Loaded", len(pieces), "pieces")
        if len(pieces) < 4:
            print("Need 4 pieces")
            continue

        start_idx, (_, _, _, _), thr = choose_first_piece_top_left(pieces)
        print("Corner-based start (top-left) piece:", pieces[start_idx]["edge_path"])
        print("Threshold for 'bad' side:", thr)

        (TL, TR, BL, BR), best_err = optimize_2x2(pieces)

        print("Best error:", best_err)
        print("Best arrangement:")
        print([p["edge_path"] for p in [TL, TR, BL, BR]])

        full_img = visualize_2x2_original(TL, TR, BL, BR, show=False)
        save_path = os.path.join(output_folder, f"puzzle_{i}.png")
        cv2.imwrite(save_path, full_img)

        print("Saved:", save_path)
