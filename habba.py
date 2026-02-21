# from pxr import UsdPhysics
# import omni.usd

# stage = omni.usd.get_context().get_stage()

# # The collision mesh is typically a child of the deformable prim
# # e.g. /World/MyDeformable/collision_mesh
# collision_mesh_prim = stage.GetPrimAtPath("/World/apple/collision_mesh")

# # ✅ Correct: PhysxSchema.PhysxCollisionAPI (NOT UsdPhysics.PhysxCollisionAPI)
# if not collision_mesh_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
#     physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(collision_mesh_prim)
# else:
#     physx_collision_api = PhysxSchema.PhysxCollisionAPI(collision_mesh_prim)

# # Set contact and rest offsets (in meters)
# physx_collision_api.GetContactOffsetAttr().Set(0.001)   # 1mm
# physx_collision_api.GetRestOffsetAttr().Set(0.0005)     # 0.5mm

import cv2

def main():
    # Open default camera (0 = primary webcam)
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    print("Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Show the frame
        cv2.imshow("Camera Feed", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()