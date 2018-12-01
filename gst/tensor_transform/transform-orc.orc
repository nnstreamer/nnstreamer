.function nns_orc_add_s8
.dest 1 d1 gint8
.source 1 s1 gint8

addssb d1, d1, s1


.function nns_orc_add_u8
.dest 1 d1 guint8
.source 1 s1 guint8

addusb d1, d1, s1


.function nns_orc_add_s16
.dest 2 d1 gint16
.source 2 s1 gint16

addssw d1, d1, s1


.function nns_orc_add_u16
.dest 2 d1 guint16
.source 2 s1 guint16

addusw d1, d1, s1


.function nns_orc_add_s32
.dest 4 d1 gint32
.source 4 s1 gint32

addssl d1, d1, s1


.function nns_orc_add_u32
.dest 4 d1 guint32
.source 4 s1 guint32

addusl d1, d1, s1


.function nns_orc_add_f32
.dest 4 d1 float
.source 4 s1 float

addf d1, d1, s1


.function nns_orc_add_f64
.dest 8 d1 double
.source 8 s1 double

addd d1, d1, s1
