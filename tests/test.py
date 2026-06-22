"""
DM Corner Diagnostic
Run this on a rotated capture to see what pylibdmtx actually reports.
Usage: python test.py <image_path>
"""

import sys
import cv2
import numpy as np
from pylibdmtx.pylibdmtx import decode

image_path = sys.argv[1]
image = cv2.imread(image_path)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
h, w = gray.shape

print(f"Image size: {w}x{h}")
print()

# Try shrink=1 first, fall back to shrink=2
for shrink in [1, 2]:
    results = decode(gray, shrink=shrink)
    if len(results) >= 2:
        print(f"Found {len(results)} DMs at shrink={shrink}")
        break
else:
    print("Could not find 2 DMs")
    sys.exit(1)

vis = image.copy()

for i, res in enumerate(results[:2]):
    r = res.rect
    data = res.data.decode('utf-8').strip()
    
    print(f"--- DM {i+1} ---")
    print(f"  data:   {data}")
    print(f"  rect:   left={r.left} top={r.top} width={r.width} height={r.height}")
    print(f"  (top is y-UP from bottom of image)")
    
    # Convert to OpenCV y-down coordinates
    scale = shrink
    left   = r.left   * scale
    top_yup = r.top   * scale
    width  = abs(r.width)  * scale
    height = abs(r.height) * scale
    
    # y-up to y-down conversion
    if r.height > 0:
        y_top_cv = h - (top_yup + r.height * scale)
    else:
        y_top_cv = h - top_yup
    
    y_bottom_cv = y_top_cv + height
    x_left      = left
    x_right     = left + width
    
    print(f"  OpenCV bbox: x=[{x_left:.0f}:{x_right:.0f}] y=[{y_top_cv:.0f}:{y_bottom_cv:.0f}]")
    print(f"  bbox size:   {width:.0f}w x {height:.0f}h")
    print(f"  aspect:      {'landscape' if width > height else 'portrait'}")
    
    # Check if pylibdmtx provides polygon/corner data beyond the rect
    print(f"  res attributes: {[a for a in dir(res) if not a.startswith('_')]}")
    
    # Draw the axis-aligned bounding box on the image
    color = (0, 255, 0) if i == 0 else (0, 0, 255)
    label = f"DM{i+1}: {data}"
    cv2.rectangle(vis,
                  (int(x_left), int(y_top_cv)),
                  (int(x_right), int(y_bottom_cv)),
                  color, 3)
    
    # Draw and number the 4 bbox corners
    corners = [
        (int(x_left),  int(y_top_cv)),     # TL
        (int(x_right), int(y_top_cv)),     # TR
        (int(x_right), int(y_bottom_cv)),  # BR
        (int(x_left),  int(y_bottom_cv)),  # BL
    ]
    corner_names = ["TL", "TR", "BR", "BL"]
    for j, (cx, cy) in enumerate(corners):
        cv2.circle(vis, (cx, cy), 12, color, -1)
        cv2.putText(vis, corner_names[j], (cx+5, cy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    # Draw center
    cx_center = int((x_left + x_right) / 2)
    cy_center = int((y_top_cv + y_bottom_cv) / 2)
    cv2.circle(vis, (cx_center, cy_center), 8, (255, 255, 0), -1)
    cv2.putText(vis, label, (int(x_left), int(y_top_cv) - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    print()

# Save output
out_path = image_path.replace(".jpeg", "_dm_corners.jpg").replace(".jpg", "_dm_corners.jpg").replace(".png", "_dm_corners.png")
# Downscale for viewing if image is very large
scale_view = min(1.0, 1200 / max(h, w))
if scale_view < 1.0:
    vis_small = cv2.resize(vis, (int(w * scale_view), int(h * scale_view)))
else:
    vis_small = vis
cv2.imwrite(out_path, vis_small)
print(f"Saved visualization to: {out_path}")
print()
print("KEY QUESTION: Do the drawn corners match the actual physical")
print("corners of the printed DM in the photo, or are they axis-aligned")
print("bounding boxes that don't follow the DM rotation?")