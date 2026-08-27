import cv2
import math
import numpy as np

img = cv2.imread('unificado_planta_h8_00.png', cv2.IMREAD_UNCHANGED)

print(img.shape)

gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY) if len(img.shape) == 3 and img.shape[2] == 4 else img
gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY) if len(gray.shape) == 3 else gray
_, gray = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)

contours, hierarchy = cv2.findContours(
    gray,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

resultado = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def eh_linha(cnt, w, h):
    if max(w, h) < 20:
        return False
    if min(w, h) <= 4 and max(w, h) / max(1, min(w, h)) > 5:
        return True
    pts = cnt.reshape(-1, 2).astype("float32")
    vx, vy, x0, y0 = cv2.fitLine(pts, cv2.DIST_L2, 0, 0.01, 0.01).flatten()
    dist = np.abs(vy * (pts[:, 0] - x0) - vx * (pts[:, 1] - y0))
    return np.percentile(dist, 90) <= max(4, max(w, h) * 0.08)


def eh_retangulo_aberto(cnt, x, y, w, h):
    if min(w, h) < 30:
        return False

    roi = np.zeros((h + 4, w + 4), dtype=np.uint8)
    local = cnt - [x - 2, y - 2]
    cv2.drawContours(roi, [local], -1, 255, 2)
    linhas = cv2.HoughLinesP(roi, 1, np.pi / 180, 25, minLineLength=min(w, h) * 0.35, maxLineGap=8)
    if linhas is None:
        return False

    horizontais = verticais = 0
    for linha in linhas:
        x1, y1, x2, y2 = linha.flatten()
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx > dy * 4:
            horizontais += 1
        elif dy > dx * 4:
            verticais += 1

    return horizontais >= 1 and verticais >= 1 and horizontais + verticais >= 3


for cnt in contours:
    area = cv2.contourArea(cnt)

    if area > 80:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)
        pontas = len(approx)

        x,y,w,h = cv2.boundingRect(cnt)
        preenchimento = area / float(w * h)
        circularidade = 4 * math.pi * area / (peri * peri) if peri else 0
        aspect_ratio = w / float(h)
        fechada = "FECHADA" if preenchimento > 0.35 else "ABERTA"


        if eh_linha(cnt, w, h):
            forma = "LINHA"
            fechada = "ABERTA"
        elif pontas == 3:
            forma = "TRIANGULO"
        elif pontas == 4 or preenchimento > 0.85:
            forma = "QUADRADO" if 0.90 <= aspect_ratio <= 1.10 else "RETANGULO"
        elif eh_retangulo_aberto(cnt, x, y, w, h):
            forma = "QUADRADO" if 0.90 <= aspect_ratio <= 1.10 else "RETANGULO"
            fechada = "ABERTA"
        elif 0.65 <= circularidade <= 1.15 and 0.75 <= aspect_ratio <= 1.25 and 0.55 <= preenchimento <= 0.85:
            forma = "CIRCULO"
        else:
            forma = "INDEFINIDA"

        cv2.drawContours(resultado, [approx], -1, (0,255,0), 2)
        cv2.rectangle(resultado, (x,y),(x+w + 2, y+h + 2), (255,0,0), 2)
        cv2.putText(resultado, f"{forma} {fechada}", (x, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        print(f"area: {area:.0f} | preenchimento: {preenchimento:.2f} | circularidade: {circularidade:.2f} | perimetro: {peri:.0f} | pontas: {pontas} | forma: {forma} | {fechada}")

linhas = cv2.HoughLinesP(gray, 1, np.pi / 180, 35, minLineLength=25, maxLineGap=5)
if linhas is not None:
    for linha in linhas:
        x1, y1, x2, y2 = linha.flatten()
        cv2.line(resultado, (x1, y1), (x2, y2), (0, 255, 255), 2)
        print(f"linha: ({x1},{y1}) -> ({x2},{y2})")

cv2.namedWindow("contornos", cv2.WINDOW_NORMAL)
cv2.imshow("contornos", resultado)
cv2.waitKey(0)
cv2.destroyAllWindows()
