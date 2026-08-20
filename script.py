import cv2

img = cv2.imread('unificado_planta_h8_00.png', cv2.IMREAD_UNCHANGED)

print(img.shape)

img = cv2.resize(img, None, fx=0.5, fy=0.5)

contours, hierarchy = cv2.findContours(
    img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

resultado = img.cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    cv2.drawContours(resultado, [cnt], -1, (255, 0, 0), 0)
    x,y,w,h = cv2.boundingRect(cnt)
    cv2.rectangle(img, (x,y),(x+w, y+h), (255,0,0), 2)

cv2.imshow("contornos", img)
cv2.waitKey(0)
cv2.destroyAllWindows()