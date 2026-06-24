import cv2
import cdp_engine
import numpy as np

image = cv2.imread(r"C:\Users\kanis\OneDrive\Desktop\AIB Innovations\QR\Data Matrix\Production\tests\batch_test\tc-03.jpeg")

from pylibdmtx.pylibdmtx import decode as dm_decode
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
raw_results = dm_decode(gray, shrink=1)

result = cdp_engine.detect_and_crop_pattern(
    image,
    dm_results_raw=raw_results,
    dm_shrink_used=1,
    seed=1079726699,
)

# Draw crop quad on full image
debug_img = image.copy()
if result[3] is not None:
    pts = result[3].astype(np.int32).reshape((-1,1,2))
    cv2.polylines(debug_img, [pts], True, (0,255,0), 8)
    for i, pt in enumerate(result[3].astype(np.int32)):
        cv2.putText(debug_img, f"{'TL TR BR BL'.split()[i]}", 
                    tuple(pt), cv2.FONT_HERSHEY_SIMPLEX, 2, (0,0,255), 4)

# Also draw DM centers
from pylibdmtx.pylibdmtx import decode as dm_decode
img_h = image.shape[0]
for r in raw_results:
    cx = int(r.rect.left + abs(r.rect.width)/2)
    cy = int(img_h - r.rect.top - abs(r.rect.height)/2)
    cv2.circle(debug_img, (cx, cy), 20, (255,0,0), -1)
    cv2.putText(debug_img, r.data.decode(), (cx+25, cy), 
                cv2.FONT_HERSHEY_SIMPLEX, 2, (255,0,0), 4)

# Save scaled down
scale_vis = 0.25
vis = cv2.resize(debug_img, None, fx=scale_vis, fy=scale_vis)
cv2.imwrite(r"C:\Users\kanis\OneDrive\Desktop\AIB Innovations\QR\Data Matrix\Production\tests\batch_test_results\tc03_debug_overlay.png", vis)
print("Saved overlay image")