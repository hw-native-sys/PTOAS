@@
 template <typename FnA2A3, typename FnA5>
 static LogicalResult dispatchVerifierByArch(Operation *op, FnA2A3 &&verifyA2A3,
                                             FnA5 &&verifyA5) {
-  switch (getVerifierTargetArch(op)) {
-  case VerifierTargetArch::A2A3:
-    return verifyA2A3();
-  case VerifierTargetArch::A5:
-    return verifyA5();
-  }
-  return failure();
+  switch (getVerifierTargetArch(op)) {
+  case VerifierTargetArch::A2A3:
+    return verifyA2A3();
+  case VerifierTargetArch::A5:
+  case VerifierTargetArch::A6: // handle A6 explicitly to avoid -Wswitch warnings
+    return verifyA5();
+  }
+  return failure();
 }
@@
-static LogicalResult verifyNamedSyncEventOp(Operation *op, PipeAttr pipe,
-                                           IntegerAttr idAttr, Value event,
-                                           int64_t expected, StringRef name)
+// Mark maybe-unused to avoid -Wunused-function when the helper isn't referenced.
+// If your project uses a pre-C++17 standard, replace with __attribute__((unused)).
+[[maybe_unused]] static LogicalResult verifyNamedSyncEventOp(Operation *op, PipeAttr pipe,
+                                                           IntegerAttr idAttr, Value event,
+                                                           int64_t expected, StringRef name)
 {
     // existing function body...
 }
