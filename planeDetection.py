# planes_from_pointcloud_exact.py
import argparse, json, math, random
from pathlib import Path
import numpy as np
import open3d as o3d

# ------------- IO -------------
def load_points(json_path: Path) -> np.ndarray:
    with open(json_path, "r") as f:
        data = json.load(f)
    pts = np.asarray(data["vertices_3d"], dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("vertices_3d must be N×3")
    return pts

# ------------- helpers -------------
def estimate_normals(pcd: o3d.geometry.PointCloud, knn: int = 30):
    pcd.estimate_normals(o3d.geometry.KDTreeSearchParamKNN(knn=knn))
    try:
        pcd.orient_normals_consistent_tangent_plane(50)
    except Exception:
        pcd.orient_normals_towards_camera_location(np.zeros(3))

def unit(v):
    n = np.linalg.norm(v) + 1e-12
    return v / n

def angle_deg(v1, v2):
    v1 = unit(v1); v2 = unit(v2)
    return math.degrees(math.acos(np.clip(np.dot(v1, v2), -1.0, 1.0)))

def segment_plane(pcd, dist, ransac_n, iters):
    if len(pcd.points) < max(200, ransac_n + 1):
        return None
    model, inliers = pcd.segment_plane(distance_threshold=dist, ransac_n=ransac_n, num_iterations=iters)
    a, b, c, d = model
    n = np.array([a, b, c], float)
    return n, float(d), np.array(inliers, int)

def best_fit_plane(points):
    c = points.mean(axis=0)
    _, _, Vt = np.linalg.svd(points - c, full_matrices=False)
    n = unit(Vt[-1])
    d = -np.dot(n, c)
    return n, c, d

def plane_frame(n):
    n = unit(n)
    a = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(a, n)) > 0.9:
        a = np.array([1.0, 0.0, 0.0])
    u = unit(np.cross(n, a))
    v = unit(np.cross(n, u))
    return u, v, n

def project_to_plane(points, p0, u, v):
    t = points - p0
    x = t @ u
    y = t @ v
    proj3d = p0 + np.outer(x, u) + np.outer(y, v)
    return np.stack([x, y], 1), proj3d

# ------------- geometry: concave hull + simplify + triangulate -------------
def rdp_simplify(poly2d, eps):
    # Ramer–Douglas–Peucker for closed polygon
    if len(poly2d) < 4:
        return poly2d
    pts = np.vstack([poly2d, poly2d[0]])
    keep = np.zeros(len(pts), dtype=bool)
    keep[0] = True; keep[-1] = True
    stack = [(0, len(pts)-1)]
    def dist_p_to_seg(p, a, b):
        ap = p - a; ab = b - a
        t = np.clip(np.dot(ap, ab) / (np.dot(ab, ab) + 1e-12), 0.0, 1.0)
        proj = a + t * ab
        return np.linalg.norm(p - proj)
    while stack:
        i, j = stack.pop()
        if j <= i + 1: continue
        a, b = pts[i], pts[j]
        idx, dmax = None, -1
        for k in range(i+1, j):
            d = dist_p_to_seg(pts[k], a, b)
            if d > dmax:
                dmax, idx = d, k
        if dmax > eps:
            keep[idx] = True
            stack.append((i, idx)); stack.append((idx, j))
    simp = pts[keep]
    return simp[:-1]  # drop duplicated start

def seg_intersect(a,b,c,d):
    # segment ab intersects cd
    def orient(p,q,r):
        return np.cross(q-p, r-p)
    def on_seg(p,q,r):
        return (min(p[0],r[0]) - 1e-12 <= q[0] <= max(p[0],r[0]) + 1e-12 and
                min(p[1],r[1]) - 1e-12 <= q[1] <= max(p[1],r[1]) + 1e-12)
    o1 = np.sign(orient(a,b,c))
    o2 = np.sign(orient(a,b,d))
    o3 = np.sign(orient(c,d,a))
    o4 = np.sign(orient(c,d,b))
    if o1 != o2 and o3 != o4: return True
    # collinear cases
    if o1 == 0 and on_seg(a,c,b): return True
    if o2 == 0 and on_seg(a,d,b): return True
    if o3 == 0 and on_seg(c,a,d): return True
    if o4 == 0 and on_seg(c,b,d): return True
    return False

def knn_concave_hull(points2d, k=20, k_max=None):
    # Moreira–Santos kNN concave hull, simple implementation
    pts = np.unique(points2d, axis=0)
    if len(pts) < 4:
        return pts
    if k_max is None: k_max = max(40, int(0.25*len(pts)))
    k = max(3, min(k, len(pts)-1))
    start = pts[np.lexsort((pts[:,0], pts[:,1]))][0]
    hull = [start]
    curr = start
    prev_dir = np.array([-1.0, 0.0])  # fire to the left initially
    used = {tuple(start)}
    for _ in range(100000):
        # distances
        d2 = np.sum((pts - curr)**2, axis=1)
        order = np.argsort(d2)
        cand = [pts[i] for i in order[1:k+1] if tuple(pts[i]) not in used or np.allclose(pts[i], start)]
        if not cand:
            k = min(k+1, k_max)
            if k == k_max: break
            continue
        # choose by minimal right turn angle > 0
        best = None
        best_ang = 1e9
        for p in cand:
            v = p - curr
            if np.linalg.norm(v) < 1e-12: continue
            v = v / np.linalg.norm(v)
            ang = math.atan2(np.cross(prev_dir, v), np.dot(prev_dir, v))
            # want smallest positive angle, wrap
            if ang <= 0: ang += 2*math.pi
            # test intersections with existing edges
            ok = True
            for i in range(len(hull)-1):
                a, b = hull[i], hull[i+1]
                if seg_intersect(a,b,curr,p):
                    ok = False; break
            if ok and len(hull) > 2 and seg_intersect(hull[0], hull[1], curr, p):
                ok = False
            if ok and ang < best_ang:
                best_ang, best = ang, p
        if best is None:
            k = min(k+1, k_max)
            if k == k_max: break
            continue
        hull.append(best)
        prev_dir = unit(best - curr)
        curr = best
        used.add(tuple(best))
        if np.allclose(curr, start) and len(hull) > 3:
            break
    hull = np.array(hull[:-1] if np.allclose(hull[-1], start) else hull, float)
    if len(hull) < 3:
        # fallback to convex hull
        return convex_hull_2d(points2d)
    return hull

def convex_hull_2d(points_2d):
    pts = np.unique(points_2d, axis=0)
    if len(pts) <= 2:
        return pts
    pts = pts[np.lexsort((pts[:,1], pts[:,0]))]
    def cross(o,a,b): return (a[0]-o[0])*(b[1]-o[1]) - (a[1]-o[1])*(b[0]-o[0])
    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(tuple(p))
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(tuple(p))
    hull = np.array(lower[:-1] + upper[:-1], float)
    return hull

def earclip_triangulate(poly2d):
    # poly2d: simple polygon, counter-clockwise preferred
    def area2(p,q,r): return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def is_ccw(poly):
        s = 0.0
        for i in range(len(poly)):
            j = (i+1) % len(poly)
            s += poly[i][0]*poly[j][1] - poly[j][0]*poly[i][1]
        return s > 0
    poly = poly2d.copy()
    if not is_ccw(poly): poly = poly[::-1]
    n = len(poly)
    idxs = list(range(n))
    tris = []
    safe = 0
    while len(idxs) > 2 and safe < 10000:
        safe += 1
        ear_found = False
        for ii in range(len(idxs)):
            i0 = idxs[(ii-1) % len(idxs)]
            i1 = idxs[ii]
            i2 = idxs[(ii+1) % len(idxs)]
            p0, p1, p2 = poly[i0], poly[i1], poly[i2]
            if area2(p0, p1, p2) <= 0:
                continue
            # point-in-triangle test for other vertices
            ok = True
            A = area2(p0, p1, p2)
            for j in idxs:
                if j in (i0, i1, i2): continue
                pj = poly[j]
                a0 = area2(p0, p1, pj)
                a1 = area2(p1, p2, pj)
                a2 = area2(p2, p0, pj)
                if a0 > 0 and a1 > 0 and a2 > 0 and (a0 + a1 + a2) <= A + 1e-9:
                    ok = False; break
            if not ok: continue
            tris.append([i0, i1, i2])
            del idxs[ii]
            ear_found = True
            break
        if not ear_found:
            # give up to avoid infinite loop
            break
    return np.array(tris, int)

# ------------- furniture suppression -------------
def remove_furniture(pcd, up, dist, iters,
                     clutter_min=0.02, clutter_max=0.40,
                     max_small_horiz_area=8.0, max_small_horiz_height=1.2,
                     normal_tol_deg=25.0, dbscan_eps=0.04, min_points=80):
    # rough floor
    work = pcd
    best = None
    for _ in range(8):
        hit = segment_plane(work, dist, 3, iters)
        if hit is None: break
        n, d, idx = hit
        if angle_deg(n, up) <= 15.0:
            if best is None or len(idx) > len(best[2]):
                best = (n, d, idx)
        work = work.select_by_index(idx, invert=True)
        if len(work.points) < 1000: break
    if best is None:
        return pcd, None
    n_floor, d_floor, idx_floor = best

    pts = np.asarray(pcd.points)
    dist_to_floor = (pts @ n_floor + d_floor) / (np.linalg.norm(n_floor) + 1e-12)

    # drop near-floor clutter band
    keep = np.ones(len(pts), bool)
    near = (dist_to_floor > clutter_min) & (dist_to_floor < clutter_max)
    keep[near] = False
    keep[idx_floor] = True

    # remove small horizontal planes above floor
    tmp = pcd.select_by_index(np.where(keep)[0])
    while True:
        hit = segment_plane(tmp, dist, 3, iters)
        if hit is None: break
        n, d, idx = hit
        ang = angle_deg(n, up)
        if ang <= 15.0:
            gidx = np.where(keep)[0][idx]
            plane_pts = pts[gidx]
            n_fit, p0, _ = best_fit_plane(plane_pts)
            u, v, _ = plane_frame(n_fit)
            xy, _ = project_to_plane(plane_pts, p0, u, v)
            hull = convex_hull_2d(xy)
            if len(hull) >= 3:
                hull3d = p0 + np.outer(hull[:,0], u) + np.outer(hull[:,1], v)
                # area in m^2
                area = polygon_area_3d(hull3d, n_fit)
            else:
                area = 0.0
            h = np.median((plane_pts @ n_floor + d_floor) / (np.linalg.norm(n_floor) + 1e-12))
            if area <= max_small_horiz_area and 0.2 <= h <= max_small_horiz_height:
                keep[gidx] = False
                tmp = pcd.select_by_index(np.where(keep)[0])
                continue
        tmp = tmp.select_by_index(idx, invert=True)
        if len(tmp.points) < 1000: break

    # normals filter: keep only near horizontal or near vertical
    tmp = pcd.select_by_index(np.where(keep)[0])
    estimate_normals(tmp, 30)
    nh = np.asarray(tmp.normals)
    horiz = np.abs(nh @ unit(up)) >= np.cos(np.deg2rad(normal_tol_deg))
    vert  = np.abs(nh @ unit(up)) <= np.cos(np.deg2rad(90 - normal_tol_deg))
    good = horiz | vert
    keep2 = np.where(keep)[0][good]

    clean = pcd.select_by_index(keep2)
    if len(clean.points) > 1000:
        labels = np.array(clean.cluster_dbscan(eps=dbscan_eps, min_points=min_points, print_progress=False))
        mask = labels >= 0
        clean_idx = keep2[mask]
        clean = pcd.select_by_index(clean_idx)
        floor_keep = np.intersect1d(idx_floor, clean_idx)
        return clean, floor_keep
    else:
        floor_keep = np.intersect1d(idx_floor, keep2)
        return clean, floor_keep

def polygon_area_3d(poly3d, n_dir):
    n_dir = unit(n_dir)
    u, v, _ = plane_frame(n_dir)
    p0 = poly3d[0]
    xy, _ = project_to_plane(poly3d, p0, u, v)
    x, y = xy[:,0], xy[:,1]
    area = 0.0
    for i in range(len(x)):
        j = (i+1) % len(x)
        area += x[i]*y[j] - x[j]*y[i]
    return abs(area) * 0.5

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("json_path", type=Path)
    ap.add_argument("--voxel", type=float, default=0.02)
    ap.add_argument("--dist", type=float, default=0.01)
    ap.add_argument("--iters", type=int, default=4000)
    ap.add_argument("--floor_max_tilt", type=float, default=10.0)
    ap.add_argument("--wall_tol", type=float, default=10.0)
    ap.add_argument("--min_cluster", type=int, default=600)
    ap.add_argument("--hull_k", type=int, default=25, help="k for kNN concave hull")
    ap.add_argument("--simplify_eps", type=float, default=0.01, help="polygon simplification, meters")
    ap.add_argument("--out", type=Path, default=Path("room_planes.obj"))
    args = ap.parse_args()

    # load and prefilter
    pts = load_points(args.json_path)
    pcd0 = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pts))
    pcd, _ = pcd0.remove_statistical_outlier(nb_neighbors=30, std_ratio=2.0)
    if args.voxel > 0:
        pcd = pcd.voxel_down_sample(args.voxel)
    estimate_normals(pcd, 30)

    up = np.array([0.0, 0.0, 1.0])
    clean_pcd, floor_idx_hint = remove_furniture(
        pcd, up, dist=args.dist, iters=args.iters
    )
    estimate_normals(clean_pcd, 30)

    # floor detection on cleaned
    floor_indices = None
    work = clean_pcd
    work_map = np.arange(len(clean_pcd.points))
    for _ in range(8):
        hit = segment_plane(work, args.dist, 3, args.iters)
        if hit is None: break
        n, d, idx = hit
        if angle_deg(n, up) <= args.floor_max_tilt:
            gidx = work_map[idx]
            if floor_indices is None or len(gidx) > len(floor_indices):
                floor_indices = gidx
        mask = np.ones(len(work_map), bool); mask[idx] = False
        work_map = work_map[mask]; work = work.select_by_index(idx, invert=True)
        if len(work.points) < args.min_cluster: break

    # walls on cleaned, floor-less cloud
    if floor_indices is not None:
        rest = clean_pcd.select_by_index(floor_indices, invert=True)
        rest_map = np.array([i for i in range(len(clean_pcd.points)) if i not in set(floor_indices)], int)
    else:
        rest = clean_pcd
        rest_map = np.arange(len(clean_pcd.points))

    walls_idx = []
    while True:
        hit = segment_plane(rest, args.dist, 3, args.iters)
        if hit is None: break
        n, d, idx = hit
        ang = angle_deg(n, up)
        gidx = rest_map[idx]
        if abs(ang - 90.0) <= args.wall_tol and len(gidx) >= args.min_cluster:
            walls_idx.append((n, gidx))
        mask = np.ones(len(rest_map), bool); mask[idx] = False
        rest_map = rest_map[mask]; rest = rest.select_by_index(idx, invert=True)
        if len(rest.points) < args.min_cluster: break

    # build concave polygons and triangulate
    meshes = []
    overlays = []
    colors = np.full((len(clean_pcd.points), 3), [0.85,0.85,0.85], float)

    def polygonize_from_inliers(idx, debug_color=(0,0,0)):
        pts3d = np.asarray(clean_pcd.points)[idx]
        n_fit, p0, _ = best_fit_plane(pts3d)
        u, v, _ = plane_frame(n_fit)
        xy, proj3d = project_to_plane(pts3d, p0, u, v)
        # concave hull around inliers
        k = min(max(10, args.hull_k), max(3, len(xy) - 1))
        hull2d = knn_concave_hull(xy, k=k)
        if len(hull2d) < 3:
            hull2d = convex_hull_2d(xy)
        # simplify in 2D
        if args.simplify_eps > 0:
            hull2d = rdp_simplify(hull2d, args.simplify_eps / max(np.linalg.norm(u), 1.0))
        # lift to 3D
        hull3d = p0 + np.outer(hull2d[:,0], u) + np.outer(hull2d[:,1], v)
        # triangulate with ear clipping
        tris2d = earclip_triangulate(hull2d)
        # build mesh
        m = o3d.geometry.TriangleMesh()
        m.vertices = o3d.utility.Vector3dVector(hull3d)
        m.triangles = o3d.utility.Vector3iVector(tris2d)
        m.compute_vertex_normals()
        return m, hull3d

    # floor mesh
    if floor_indices is not None and len(floor_indices) > 0:
        colors[floor_indices] = np.array([0.1,0.9,0.1])
        floor_mesh, floor_poly = polygonize_from_inliers(floor_indices)
        floor_mesh.paint_uniform_color([0.2, 0.7, 0.2])
        meshes.append(floor_mesh)
        # overlay
        ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(floor_poly),
                                  lines=o3d.utility.Vector2iVector(np.array([[i,(i+1)%len(floor_poly)] for i in range(len(floor_poly))],int)))
        ls.colors = o3d.utility.Vector3dVector(np.tile([0,0,0], (len(floor_poly),1)))
        overlays.append(ls)

    # wall meshes
    for n_detect, idx in walls_idx:
        colors[idx] = np.array([random.uniform(0.2,0.95), random.uniform(0.2,0.95), random.uniform(0.2,0.95)])
        wall_mesh, wall_poly = polygonize_from_inliers(idx)
        wall_mesh.paint_uniform_color([0.65, 0.65, 0.85])
        meshes.append(wall_mesh)
        ls = o3d.geometry.LineSet(points=o3d.utility.Vector3dVector(wall_poly),
                                  lines=o3d.utility.Vector2iVector(np.array([[i,(i+1)%len(wall_poly)] for i in range(len(wall_poly))],int)))
        ls.colors = o3d.utility.Vector3dVector(np.tile([0,0,0], (len(wall_poly),1)))
        overlays.append(ls)

    # export
    if meshes:
        combined = o3d.geometry.TriangleMesh()
        voff = 0
        vs, fs, cs = [], [], []
        for m in meshes:
            v = np.asarray(m.vertices); f = np.asarray(m.triangles)
            vs.append(v); fs.append(f + voff); voff += len(v)
        combined.vertices = o3d.utility.Vector3dVector(np.vstack(vs))
        combined.triangles = o3d.utility.Vector3iVector(np.vstack(fs))
        combined.compute_vertex_normals()
        o3d.io.write_triangle_mesh(str(args.out), combined)
        print(f"Exported planes to {args.out.resolve()}")

    # show
    clean_pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([clean_pcd] + meshes + overlays,
                                      window_name="Exact planes from inliers (concave hull + triangulation)")

if __name__ == "__main__":
    main()
