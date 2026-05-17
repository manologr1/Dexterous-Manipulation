import hppfcl
import pinocchio as pin


def addSystemCollisionPairs(model, geom_model, qref):
    """
    Add the right collision pairs of a model, given qref.
    qref is here as a `T-pose`. The function uses this pose to determine which objects are in collision
    in this ref pose. If objects are in collision, they are not added as collision pairs, as they are considered
    to always be in collision.
    """
    data = model.createData()
    geom_data = geom_model.createData()
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, qref)
    geom_model.removeAllCollisionPairs()
    num_col_pairs = 0
    for i in range(len(geom_model.geometryObjects)):
        for j in range(i + 1, len(geom_model.geometryObjects)):
            # Don't add collision pair if same object
            if i != j:
                gobj_i: pin.GeometryObject = geom_model.geometryObjects[i]
                gobj_j: pin.GeometryObject = geom_model.geometryObjects[j]
                if gobj_i.name == "floor" or gobj_j.name == "floor":
                    num_col_pairs += 1
                    col_pair = pin.CollisionPair(i, j)
                    geom_model.addCollisionPair(col_pair)
                else:
                    if (
                        gobj_i.parentJoint != gobj_j.parentJoint
                        or gobj_i.parentJoint == 0
                    ):
                        if (
                            gobj_i.parentJoint != model.parents[gobj_j.parentJoint]
                            and gobj_j.parentJoint != model.parents[gobj_i.parentJoint]
                            or gobj_i.parentJoint == 0
                            or gobj_j.parentJoint == 0
                        ):
                            # Compute collision between the geometries. Only add the collision pair if there is no collision.
                            M1 = geom_data.oMg[i]
                            M2 = geom_data.oMg[j]
                            colreq = hppfcl.CollisionRequest()
                            colreq.security_margin = 1e-2  # 1cm of clearance
                            colres = hppfcl.CollisionResult()
                            hppfcl.collide(
                                gobj_i.geometry, M1, gobj_j.geometry, M2, colreq, colres
                            )
                            if not colres.isCollision():
                                num_col_pairs += 1
                                col_pair = pin.CollisionPair(i, j)
                                geom_model.addCollisionPair(col_pair)
    print("Num col pairs = ", num_col_pairs)


def addSelectedCollisionPairs(model, geom_model, qref, include_geom_substrings):
    """
    Like addSystemCollisionPairs but only considers pairs where BOTH geometries
    match one of the substrings in include_geom_substrings.
    All other filters from the original (same-joint, adjacent-joint,
    already-in-collision-at-qref, floor handling) are preserved.
    """
    data = model.createData()
    geom_data = geom_model.createData()
    pin.updateGeometryPlacements(model, data, geom_model, geom_data, qref)
    geom_model.removeAllCollisionPairs()

    def _matches(name):
        return any(s in name for s in include_geom_substrings)

    # Build the whitelisted index set once
    relevant_ids = [
        i for i, g in enumerate(geom_model.geometryObjects)
        if _matches(g.name)
    ]
    print(f"Whitelisted geometries ({len(relevant_ids)}):")
    for i in relevant_ids:
        print(f"  {i:3d}: {geom_model.geometryObjects[i].name}")

    num_col_pairs = 0
    for ii in range(len(relevant_ids)):
        i = relevant_ids[ii]
        for jj in range(ii + 1, len(relevant_ids)):
            j = relevant_ids[jj]

            # Don't add collision pair if same object (redundant since j > i,
            # but kept for parity with the original)
            if i != j:
                gobj_i: pin.GeometryObject = geom_model.geometryObjects[i]
                gobj_j: pin.GeometryObject = geom_model.geometryObjects[j]

                if gobj_i.name == "floor" or gobj_j.name == "floor":
                    num_col_pairs += 1
                    col_pair = pin.CollisionPair(i, j)
                    geom_model.addCollisionPair(col_pair)
                else:
                    if (
                        gobj_i.parentJoint != gobj_j.parentJoint
                        or gobj_i.parentJoint == 0
                    ):
                        if (
                            gobj_i.parentJoint != model.parents[gobj_j.parentJoint]
                            and gobj_j.parentJoint != model.parents[gobj_i.parentJoint]
                            or gobj_i.parentJoint == 0
                            or gobj_j.parentJoint == 0
                        ):
                            # Compute collision between the geometries.
                            # Only add the collision pair if there is no collision.
                            M1 = geom_data.oMg[i]
                            M2 = geom_data.oMg[j]
                            colreq = hppfcl.CollisionRequest()
                            colreq.security_margin = 1e-2  # 1cm of clearance
                            colres = hppfcl.CollisionResult()
                            hppfcl.collide(
                                gobj_i.geometry, M1, gobj_j.geometry, M2, colreq, colres
                            )
                            if not colres.isCollision():
                                num_col_pairs += 1
                                col_pair = pin.CollisionPair(i, j)
                                geom_model.addCollisionPair(col_pair)
    print("Num col pairs = ", num_col_pairs)
    return num_col_pairs
