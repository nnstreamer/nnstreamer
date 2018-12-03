.function nns_orc_add_c_s8
.dest 1 d1 int8_t
.param 1 p1 int8_t

addssb d1, d1, p1


.function nns_orc_mul_c_s8
.dest 1 d1 int8_t
.param 1 p1 int8_t
.temp 2 t1

mulsbw t1, d1, p1
convssswb d1, t1


.function nns_orc_add_c_u8
.dest 1 d1 uint8_t
.param 1 p1 uint8_t

addusb d1, d1, p1


.function nns_orc_mul_c_u8
.dest 1 d1 uint8_t
.param 1 p1 uint8_t
.temp 2 t1

mulubw t1, d1, p1
convuuswb d1, t1


.function nns_orc_add_c_s16
.dest 2 d1 int16_t
.param 2 p1 int16_t

addssw d1, d1, p1


.function nns_orc_mul_c_s16
.dest 2 d1 int16_t
.param 2 p1 int16_t
.temp 4 t1

mulswl t1, d1, p1
convssslw d1, t1


.function nns_orc_add_c_u16
.dest 2 d1 uint16_t
.param 2 p1 uint16_t

addusw d1, d1, p1


.function nns_orc_mul_c_u16
.dest 2 d1 uint16_t
.param 2 p1 uint16_t
.temp 4 t1

muluwl t1, d1, p1
convuuslw d1, t1


.function nns_orc_add_c_s32
.dest 4 d1 int32_t
.param 4 p1 int32_t

addssl d1, d1, p1


.function nns_orc_mul_c_s32
.dest 4 d1 int32_t
.param 4 p1 int32_t
.temp 8 t1

mulslq t1, d1, p1
convsssql d1, t1


.function nns_orc_add_c_u32
.dest 4 d1 uint32_t
.param 4 p1 uint32_t

addusl d1, d1, p1


.function nns_orc_mul_c_u32
.dest 4 d1 uint32_t
.param 4 p1 uint32_t
.temp 8 t1

mululq t1, d1, p1
convuusql d1, t1


.function nns_orc_add_c_f32
.dest 4 d1 float
.floatparam 4 p1 float

addf d1, d1, p1


.function nns_orc_mul_c_f32
.dest 4 d1 float
.floatparam 4 p1 float

mulf d1, d1, p1


.function nns_orc_add_c_f64
.dest 8 d1 double
.doubleparam 8 p1 double

addd d1, d1, p1


.function nns_orc_mul_c_f64
.dest 8 d1 double
.doubleparam 8 p1 double

muld d1, d1, p1
