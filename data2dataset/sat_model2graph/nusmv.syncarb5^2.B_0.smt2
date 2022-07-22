; benchmark generated from python API
(set-info :status unknown)
(declare-fun v12 () Bool)
(declare-fun v24 () Bool)
(declare-fun v18 () Bool)
(declare-fun v30 () Bool)
(declare-fun v14 () Bool)
(declare-fun v22 () Bool)
(declare-fun i8 () Bool)
(declare-fun v26 () Bool)
(declare-fun i2 () Bool)
(assert
 (= v12 true))
(assert
 (= v24 true))
(assert
 (= v18 true))
(assert
 (= v30 false))
(assert
 (let (($x65 (not v24)))
 (let (($x71 (not v26)))
 (let (($x499 (and $x71 $x65)))
 (let (($x634 (not $x499)))
 (let (($x485 (and $x634 i8)))
 (let (($x450 (not v12)))
 (let (($x506 (not v14)))
 (let (($x500 (and $x506 $x450)))
 (let (($x501 (not $x500)))
 (let (($x502 (and $x501 i2)))
 ;(let (($x502 (and $x634 i8)))
 (let (($x164 (not (and v12 v24 v18 (not v30)))))
 (not (and $x164 (and $x502 $x485 v22 (not $x506))))))))))))))))
(check-sat)
