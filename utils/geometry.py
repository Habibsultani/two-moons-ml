"""İki boyutta tek doğruyla kusursuz ayrılmayı engelleyen bir geometrik tanık.

Sadece veri incelemesidir; model eğitmez. Tanık bulunamaması ayrılabilirlik kanıtı değildir.
"""

import numpy as np


def cross(a, b):
    return float(a[0] * b[1] - a[1] * b[0])


def convex_hull_indices(points):
    order = np.lexsort((points[:, 1], points[:, 0]))
    def half(indices):
        hull = []
        for i in indices:
            while len(hull) >= 2 and cross(points[hull[-1]] - points[hull[-2]],
                                           points[i] - points[hull[-2]]) <= 0:
                hull.pop()
            hull.append(int(i))
        return hull
    return half(order)[:-1] + half(order[::-1])[:-1]


def linear_nonseparability_certificate(X, y, source_rows):
    """Karşıt sınıfların dışbükey kılıf kenarlarında kesişim arar."""
    X, y, source_rows = np.asarray(X), np.asarray(y), np.asarray(source_rows)
    groups = [np.flatnonzero(y == label) for label in [0, 1]]
    hulls = [[int(group[i]) for i in convex_hull_indices(X[group])] for group in groups]
    for i, a_idx in enumerate(hulls[0]):
        b_idx = hulls[0][(i + 1) % len(hulls[0])]
        a, r = X[a_idx], X[b_idx] - X[a_idx]
        for j, c_idx in enumerate(hulls[1]):
            d_idx = hulls[1][(j + 1) % len(hulls[1])]
            c, s = X[c_idx], X[d_idx] - X[c_idx]
            denominator = cross(r, s)
            if abs(denominator) < 1e-12:
                continue
            t, u = cross(c - a, s) / denominator, cross(c - a, r) / denominator
            if 1e-9 < t < 1 - 1e-9 and 1e-9 < u < 1 - 1e-9:
                point0, point1 = a + t * r, c + u * s
                return {"found": True, "scope": f"Yalnızca {len(X)} dış eğitim örneği",
                        "class_0_rows": source_rows[[a_idx, b_idx]].tolist(),
                        "class_1_rows": source_rows[[c_idx, d_idx]].tolist(),
                        "class_0_weights": [1 - t, t], "class_1_weights": [1 - u, u],
                        "intersection": point0.tolist(),
                        "maximum_coordinate_error": float(np.max(np.abs(point0 - point1))),
                        "interpretation": "Karşıt sınıf dışbükey kılıfları kesişiyor; sıkı doğrusal ayırıcı yok.",
                        "limitation": "Doğrusal olmayan bir modelin kusursuz genelleyeceğini göstermez."}
    return {"found": False, "interpretation": "Kenar kesişimi tanığı bulunamadı; ayrılabilirlik sonucu çıkarılamaz."}
