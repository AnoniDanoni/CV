import cv2

img = cv2.imread('unificado_planta_h8_00.png', cv2.IMREAD_UNCHANGED)

print(img.shape)

img = cv2.resize(img, None, fx=0.5, fy=0.5)

contours, hierarchy = cv2.findContours(
    img,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

resultado = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 500:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        pontas = len(approx)

        x,y,w,h = cv2.boundingRect(approx)
        cv2.drawContours(resultado, [approx], -1, (0,255,0), 2)
        cv2.rectangle(resultado, (x,y),(x+w + 2, y+h + 2), (255,0,0), 2)
        cv2.putText(resultado, str(pontas), (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        if pontas == 2:
            pontas = "LINHA"
        elif pontas == 3:
            pontas = "TRIANGULO"
        elif pontas == 4:
            pontas = "QUADRADO"
        elif pontas > 4:
            pontas = "CIRCULO"
        print(f"area: {area:.0f} | perimetro: {peri:.0f} | pontas: {pontas}")

cv2.imshow("contornos", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
