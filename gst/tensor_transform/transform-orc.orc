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


.function nns_orc_conv_s8_to_s8
.dest 1 d1 int8_t
.source 1 s1 int8_t

copyb d1, s1


.function nns_orc_conv_s8_to_u8
.dest 1 d1 uint8_t
.source 1 s1 int8_t

copyb d1, s1


.function nns_orc_conv_s8_to_s16
.dest 2 d1 int16_t
.source 1 s1 int8_t

convsbw d1, s1


.function nns_orc_conv_s8_to_u16
.dest 2 d1 uint16_t
.source 1 s1 int8_t

convsbw d1, s1


.function nns_orc_conv_s8_to_s32
.dest 4 d1 int32_t
.source 1 s1 int8_t
.temp 2 t1

convsbw t1, s1
convswl d1, t1


.function nns_orc_conv_s8_to_u32
.dest 4 d1 uint32_t
.source 1 s1 int8_t
.temp 2 t1

convsbw t1, s1
convswl d1, t1


.function nns_orc_conv_s8_to_f32
.dest 4 d1 float
.source 1 s1 int8_t
.temp 2 t1
.temp 4 t2

convsbw t1, s1
convswl t2, t1
convlf d1, t2


.function nns_orc_conv_s8_to_f64
.dest 8 d1 double
.source 1 s1 int8_t
.temp 2 t1
.temp 4 t2

convsbw t1, s1
convswl t2, t1
convld d1, t2


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


.function nns_orc_conv_u8_to_s8
.dest 1 d1 int8_t
.source 1 s1 uint8_t

copyb d1, s1


.function nns_orc_conv_u8_to_u8
.dest 1 d1 uint8_t
.source 1 s1 uint8_t

copyb d1, s1


.function nns_orc_conv_u8_to_s16
.dest 2 d1 int16_t
.source 1 s1 uint8_t

convubw d1, s1


.function nns_orc_conv_u8_to_u16
.dest 2 d1 uint16_t
.source 1 s1 uint8_t

convubw d1, s1


.function nns_orc_conv_u8_to_s32
.dest 4 d1 int32_t
.source 1 s1 uint8_t
.temp 2 t1

convubw t1, s1
convuwl d1, t1


.function nns_orc_conv_u8_to_u32
.dest 4 d1 uint32_t
.source 1 s1 uint8_t
.temp 2 t1

convubw t1, s1
convuwl d1, t1


.function nns_orc_conv_u8_to_f32
.dest 4 d1 float
.source 1 s1 uint8_t
.temp 2 t1
.temp 4 t2

convubw t1, s1
convuwl t2, t1
convlf d1, t2


.function nns_orc_conv_u8_to_f64
.dest 8 d1 double
.source 1 s1 uint8_t
.temp 2 t1
.temp 4 t2

convubw t1, s1
convuwl t2, t1
convld d1, t2


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


.function nns_orc_conv_s16_to_s8
.dest 1 d1 int8_t
.source 2 s1 int16_t

convssswb d1, s1


.function nns_orc_conv_s16_to_u8
.dest 1 d1 uint8_t
.source 2 s1 int16_t

convwb d1, s1


.function nns_orc_conv_s16_to_s16
.dest 2 d1 int16_t
.source 2 s1 int16_t

copyw d1, s1


.function nns_orc_conv_s16_to_u16
.dest 2 d1 uint16_t
.source 2 s1 int16_t

copyw d1, s1


.function nns_orc_conv_s16_to_s32
.dest 4 d1 int32_t
.source 2 s1 int16_t

convswl d1, s1


.function nns_orc_conv_s16_to_u32
.dest 4 d1 uint32_t
.source 2 s1 int16_t
.temp 4 t1

convswl d1, s1


.function nns_orc_conv_s16_to_f32
.dest 4 d1 float
.source 2 s1 int16_t
.temp 4 t1

convswl t1, s1
convlf d1, t1


.function nns_orc_conv_s16_to_f64
.dest 8 d1 double
.source 2 s1 int16_t
.temp 4 t1

convswl t1, s1
convld d1, t1


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


.function nns_orc_conv_u16_to_s8
.dest 1 d1 int8_t
.source 2 s1 uint16_t

convssswb d1, s1


.function nns_orc_conv_u16_to_u8
.dest 1 d1 uint8_t
.source 2 s1 uint16_t

convwb d1, s1


.function nns_orc_conv_u16_to_s16
.dest 2 d1 int16_t
.source 2 s1 uint16_t

copyw d1, s1


.function nns_orc_conv_u16_to_u16
.dest 2 d1 uint16_t
.source 2 s1 uint16_t

copyw d1, s1


.function nns_orc_conv_u16_to_s32
.dest 4 d1 int32_t
.source 2 s1 uint16_t

convuwl d1, s1


.function nns_orc_conv_u16_to_u32
.dest 4 d1 uint32_t
.source 2 s1 uint16_t

convuwl d1, s1


.function nns_orc_conv_u16_to_f32
.dest 4 d1 float
.source 2 s1 uint16_t
.temp 4 t1

convuwl t1, s1
convlf d1, t1


.function nns_orc_conv_u16_to_f64
.dest 8 d1 double
.source 2 s1 uint16_t
.temp 4 t1

convuwl t1, s1
convld d1, t1


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


.function nns_orc_conv_s32_to_s8
.dest 1 d1 int8_t
.source 4 s1 int32_t
.temp 2 t1

convssslw t1, s1
convssswb d1, t1


.function nns_orc_conv_s32_to_u8
.dest 1 d1 uint8_t
.source 4 s1 int32_t
.temp 2 t1

convlw t1, s1
convwb d1, t1


.function nns_orc_conv_s32_to_s16
.dest 2 d1 int16_t
.source 4 s1 int32_t

convssslw d1, s1


.function nns_orc_conv_s32_to_u16
.dest 2 d1 uint16_t
.source 4 s1 int32_t
.temp 2 t1

convssslw d1, s1


.function nns_orc_conv_s32_to_s32
.dest 4 d1 int32_t
.source 4 s1 int32_t

copyl d1, s1


.function nns_orc_conv_s32_to_u32
.dest 4 d1 uint32_t
.source 4 s1 int32_t

copyl d1, s1


.function nns_orc_conv_s32_to_f32
.dest 4 d1 float
.source 4 s1 int32_t

convlf d1, s1


.function nns_orc_conv_s32_to_f64
.dest 8 d1 double
.source 4 s1 int32_t

convld d1, s1


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


.function nns_orc_conv_u32_to_s8
.dest 1 d1 int8_t
.source 4 s1 uint32_t
.temp 2 t1

convssslw t1, s1
convssswb d1, t1


.function nns_orc_conv_u32_to_u8
.dest 1 d1 uint8_t
.source 4 s1 uint32_t
.temp 2 t1

convlw t1, s1
convwb d1, t1


.function nns_orc_conv_u32_to_s16
.dest 2 d1 int16_t
.source 4 s1 uint32_t

convssslw d1, s1


.function nns_orc_conv_u32_to_u16
.dest 2 d1 uint16_t
.source 4 s1 uint32_t

convssslw d1, s1


.function nns_orc_conv_u32_to_s32
.dest 4 d1 int32_t
.source 4 s1 uint32_t

copyl d1, s1


.function nns_orc_conv_u32_to_u32
.dest 4 d1 uint32_t
.source 4 s1 uint32_t

copyl d1, s1


.function nns_orc_conv_u32_to_f32
.dest 4 d1 float
.source 4 s1 uint32_t

convlf d1, s1


.function nns_orc_conv_u32_to_f64
.dest 8 d1 double
.source 4 s1 uint32_t

convld d1, s1


.function nns_orc_add_c_f32
.dest 4 d1 float
.floatparam 4 p1 float

addf d1, d1, p1


.function nns_orc_mul_c_f32
.dest 4 d1 float
.floatparam 4 p1 float

mulf d1, d1, p1


.function nns_orc_div_c_f32
.dest 4 d1 float
.floatparam 4 p1 float

divf d1, d1, p1


.function nns_orc_conv_f32_to_s8
.dest 1 d1 int8_t
.source 4 s1 float
.temp 4 t1
.temp 2 t2

convfl t1, s1
convssslw t2, t1
convssswb d1, t2


.function nns_orc_conv_f32_to_u8
.dest 1 d1 uint8_t
.source 4 s1 float
.temp 4 t1
.temp 2 t2

convfl t1, s1
convlw t2, t1
convwb d1, t2


.function nns_orc_conv_f32_to_s16
.dest 2 d1 int16_t
.source 4 s1 float
.temp 4 t1

convfl t1, s1
convssslw d1, t1


.function nns_orc_conv_f32_to_u16
.dest 2 d1 uint16_t
.source 4 s1 float
.temp 4 t1

convfl t1, s1
convssslw d1, t1


.function nns_orc_conv_f32_to_s32
.dest 4 d1 int32_t
.source 4 s1 float

convfl d1, s1


.function nns_orc_conv_f32_to_u32
.dest 4 d1 uint32_t
.source 4 s1 float

convfl d1, s1


.function nns_orc_conv_f32_to_f32
.dest 4 d1 float
.source 4 s1 float

copyl d1, s1


.function nns_orc_conv_f32_to_f64
.dest 8 d1 double
.source 4 s1 float

convfd d1, s1


.function nns_orc_add_c_f64
.dest 8 d1 double
.doubleparam 8 p1 double

addd d1, d1, p1


.function nns_orc_mul_c_f64
.dest 8 d1 double
.doubleparam 8 p1 double

muld d1, d1, p1


.function nns_orc_div_c_f64
.dest 8 d1 double
.doubleparam 8 p1 double

divd d1, d1, p1


.function nns_orc_conv_f64_to_s8
.dest 1 d1 int8_t
.source 8 s1 double
.temp 4 t1
.temp 2 t2

convdl t1, s1
convssslw t2, t1
convssswb d1, t2


.function nns_orc_conv_f64_to_u8
.dest 1 d1 uint8_t
.source 8 s1 double
.temp 4 t1
.temp 2 t2

convdl t1, s1
convlw t2, t1
convwb d1, t2


.function nns_orc_conv_f64_to_s16
.dest 2 d1 int16_t
.source 8 s1 double
.temp 4 t1

convdl t1, s1
convssslw d1, t1


.function nns_orc_conv_f64_to_u16
.dest 2 d1 uint16_t
.source 8 s1 double
.temp 4 t1

convdl t1, s1
convssslw d1, t1


.function nns_orc_conv_f64_to_s32
.dest 4 d1 int32_t
.source 8 s1 double

convdl d1, s1


.function nns_orc_conv_f64_to_u32
.dest 4 d1 uint32_t
.source 8 s1 double

convdl d1, s1


.function nns_orc_conv_f64_to_f32
.dest 4 d1 float
.source 8 s1 double

convdf d1, s1


.function nns_orc_conv_f64_to_f64
.dest 8 d1 double
.source 8 s1 double

copyq d1, s1
