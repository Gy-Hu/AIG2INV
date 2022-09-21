; benchmark generated from python API
(set-info :status unknown)
(declare-fun v18 () Bool)
(declare-fun v14 () Bool)
(declare-fun v22 () Bool)
(declare-fun v26 () Bool)
(assert
 (= v18 true))
(assert
 (= v14 true))
(assert
 (= v22 true))
(assert
(not (and (not (and v18 v14 v22)) (and v22 v18 v26))))
;(not (and (not (and v18 v14 v22)) (not (and v18 v14 v22)))))
(check-sat)
