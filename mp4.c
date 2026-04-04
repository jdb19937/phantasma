/*
 * pfr_mp4.c — inscriptor MP4 (H.264 Baseline)
 * ===============================================
 *
 * ARGB8888 tabulas in MP4 plicam cum H.264 codice scribit.
 * I-PCM modus adhibetur (pixeli non compressi, sed validus H.264).
 * Omnis tabula est IDR (instantanea).
 *
 * Structura:
 *   ftyp → mdat (tabulae encodatae) → moov (metadata)
 */

#include "phantasma.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* ================================================================
 * bitstream scriptor — ad SPS/PPS/slice capita scribenda
 * ================================================================ */

typedef struct {
    uint8_t *data;
    int capacitas;
    int byte_pos;
    int bit_pos;    /* 0-7, bits in current byte */
} bs_t;

static void bs_initia(bs_t *bs, uint8_t *data, int cap)
{
    bs->data      = data;
    bs->capacitas = cap;
    bs->byte_pos  = 0;
    bs->bit_pos   = 0;
    memset(data, 0, (size_t)cap);
}

static void bs_u(bs_t *bs, int n, uint32_t val)
{
    for (int i = n - 1; i >= 0; i--) {
        int bit = (val >> i) & 1;
        bs->data[bs->byte_pos] |= (uint8_t)(bit << (7 - bs->bit_pos));
        bs->bit_pos++;
        if (bs->bit_pos == 8) {
            bs->bit_pos = 0;
            bs->byte_pos++;
        }
    }
}

/* Exp-Golomb unsigned */
static void bs_ue(bs_t *bs, uint32_t val)
{
    uint32_t m   = val + 1;
    int bits     = 0;
    uint32_t tmp = m;
    while (tmp > 0) {
        bits++;
        tmp >>= 1;
    }
    for (int i = 0; i < bits - 1; i++)
        bs_u(bs, 1, 0);
    bs_u(bs, bits, m);
}

/* Exp-Golomb signed */
static void bs_se(bs_t *bs, int32_t val)
{
    uint32_t mapped;
    if (val > 0)
        mapped = (uint32_t)(val * 2 - 1);
    else
        mapped = (uint32_t)(-val * 2);
    bs_ue(bs, mapped);
}

static int bs_longitudo(bs_t *bs)
{
    return bs->byte_pos + (bs->bit_pos > 0 ? 1 : 0);
}

/* RBSP trailing bits */
static void bs_rbsp_fini(bs_t *bs)
{
    bs_u(bs, 1, 1);
    while (bs->bit_pos != 0)
        bs_u(bs, 1, 0);
}

/* ================================================================
 * emulation prevention — 0x000000/01/02/03 -> 0x00000300/01/02/03
 * ================================================================ */

static int nal_emittere(uint8_t *dest, const uint8_t *rbsp, int rbsp_lon)
{
    int j     = 0;
    int zeros = 0;
    for (int i = 0; i < rbsp_lon; i++) {
        if (zeros >= 2 && rbsp[i] <= 3) {
            dest[j++] = 0x03;
            zeros     = 0;
        }
        dest[j++] = rbsp[i];
        if (rbsp[i] == 0)
            zeros++;
        else
            zeros = 0;
    }
    return j;
}

/* ================================================================
 * RGB -> YUV420 conversio
 * ================================================================ */

static void argb_ad_yuv420(
    const uint32_t *argb, int lat, int alt,
    uint8_t *y_plan, uint8_t *cb_plan,
    uint8_t *cr_plan
) {
    int lat_c = lat / 2;

    for (int py = 0; py < alt; py++) {
        for (int px = 0; px < lat; px++) {
            uint32_t px_val = argb[py * lat + px];
            int r = (px_val >> 16) & 0xFF;
            int g = (px_val >>  8) & 0xFF;
            int b =  px_val        & 0xFF;

            int yv =  ((66 * r + 129 * g +  25 * b + 128) >> 8) + 16;
            if (yv < 16)
                yv = 16;
            if (yv > 235)
                yv = 235;
            y_plan[py * lat + px] = (uint8_t)yv;
        }
    }

    for (int py = 0; py < alt; py += 2) {
        for (int px = 0; px < lat; px += 2) {
            int r = 0, g = 0, b = 0;
            for (int dy = 0; dy < 2; dy++) {
                for (int dx = 0; dx < 2; dx++) {
                    int sy = py + dy, sx = px + dx;
                    if (sy >= alt)
                        sy = alt - 1;
                    if (sx >= lat)
                        sx = lat - 1;
                    uint32_t px_val = argb[sy * lat + sx];
                    r += (px_val >> 16) & 0xFF;
                    g += (px_val >>  8) & 0xFF;
                    b +=  px_val        & 0xFF;
                }
            }
            r /= 4;
            g /= 4;
            b /= 4;

            int cb = ((-38 * r -  74 * g + 112 * b + 128) >> 8) + 128;
            int cr = ((112 * r -  94 * g -  18 * b + 128) >> 8) + 128;
            if (cb < 16)
                cb = 16;
            if (cb > 240)
                cb = 240;
            if (cr < 16)
                cr = 16;
            if (cr > 240)
                cr = 240;
            cb_plan[(py / 2) * lat_c + (px / 2)] = (uint8_t)cb;
            cr_plan[(py / 2) * lat_c + (px / 2)] = (uint8_t)cr;
        }
    }
}

/* ================================================================
 * MP4 auxiliares — big-endian scripta
 * ================================================================ */

static void mp4_u32(FILE *f, uint32_t v)
{
    fputc((v >> 24) & 0xFF, f);
    fputc((v >> 16) & 0xFF, f);
    fputc((v >>  8) & 0xFF, f);
    fputc( v        & 0xFF, f);
}

static void mp4_u16(FILE *f, uint16_t v)
{
    fputc((v >> 8) & 0xFF, f);
    fputc( v       & 0xFF, f);
}

static void mp4_u8(FILE *f, uint8_t v)
{
    fputc(v, f);
}

static void mp4_capsam_initia(FILE *f, const char *typ, long *pos)
{
    *pos = ftell(f);
    mp4_u32(f, 0);          /* placeholder magnitudinis */
    fwrite(typ, 1, 4, f);
}

static void mp4_capsam_fini(FILE *f, long pos)
{
    long nunc    = ftell(f);
    uint32_t mag = (uint32_t)(nunc - pos);
    fseek(f, pos, SEEK_SET);
    mp4_u32(f, mag);
    fseek(f, nunc, SEEK_SET);
}

static void mp4_zeros(FILE *f, int n)
{
    for (int i = 0; i < n; i++)
        fputc(0, f);
}

/* ================================================================
 * structura MP4
 * ================================================================ */

struct pfr_mp4 {
    FILE *plica;
    int latitudo, altitudo;
    int fps;
    int numerus;

    /* MB dimensiones */
    int mbs_lat, mbs_alt;

    /* SPS et PPS (NAL unitates sine emulation prevention) */
    uint8_t sps[64];
    int sps_lon;
    uint8_t pps[64];
    int pps_lon;

    /* magnitudines tabularum pro moov */
    uint32_t *magnitudines;
    int capacitas;

    /* mdat positiones */
    long mdat_pos;
    long mdat_data_pos;

    /* plana YUV */
    uint8_t *y_plan;
    uint8_t *cb_plan;
    uint8_t *cr_plan;

    /* alveus NAL */
    uint8_t *nal_alveus;
    int nal_cap;
};

/* ================================================================
 * SPS et PPS generare
 * ================================================================ */

static void sps_genera(pfr_mp4_t *m)
{
    uint8_t rbsp[64];
    bs_t bs;
    bs_initia(&bs, rbsp, sizeof(rbsp));

    /* NAL header */
    bs_u(&bs, 1, 0);       /* forbidden_zero_bit */
    bs_u(&bs, 2, 3);       /* nal_ref_idc = 3 */
    bs_u(&bs, 5, 7);       /* nal_unit_type = 7 (SPS) */

    /* SPS */
    bs_u(&bs, 8, 66);      /* profile_idc = Baseline */
    bs_u(&bs, 1, 1);       /* constraint_set0_flag */
    bs_u(&bs, 1, 1);       /* constraint_set1_flag */
    bs_u(&bs, 1, 0);       /* constraint_set2_flag */
    bs_u(&bs, 1, 0);       /* constraint_set3_flag */
    bs_u(&bs, 4, 0);       /* reserved_zero_4bits */
    bs_u(&bs, 8, 30);      /* level_idc = 3.0 */

    bs_ue(&bs, 0);         /* seq_parameter_set_id */
    bs_ue(&bs, 0);         /* log2_max_frame_num_minus4 → max_frame_num=16 */
    bs_ue(&bs, 2);         /* pic_order_cnt_type = 2 (nulla POC in slice) */
    bs_ue(&bs, 0);         /* max_num_ref_frames = 0 */
    bs_u(&bs, 1, 0);       /* gaps_in_frame_num_value_allowed_flag */

    bs_ue(&bs, (uint32_t)(m->mbs_lat - 1));  /* pic_width_in_mbs_minus1 */
    bs_ue(&bs, (uint32_t)(m->mbs_alt - 1));  /* pic_height_in_map_units_minus1 */
    bs_u(&bs, 1, 1);       /* frame_mbs_only_flag */
    /* no mb_adaptive_frame_field since frame_mbs_only=1 */
    bs_u(&bs, 1, 0);       /* direct_8x8_inference_flag */

    /* frame cropping si necessarium */
    int crop_r = m->mbs_lat * 16 - m->latitudo;
    int crop_b = m->mbs_alt * 16 - m->altitudo;
    if (crop_r > 0 || crop_b > 0) {
        bs_u(&bs, 1, 1);           /* frame_cropping_flag */
        bs_ue(&bs, 0);             /* crop_left */
        bs_ue(&bs, (uint32_t)(crop_r / 2));  /* crop_right (chroma units) */
        bs_ue(&bs, 0);             /* crop_top */
        bs_ue(&bs, (uint32_t)(crop_b / 2));  /* crop_bottom */
    } else {
        bs_u(&bs, 1, 0);
    }

    bs_u(&bs, 1, 0);       /* vui_parameters_present_flag */
    bs_rbsp_fini(&bs);

    m->sps_lon = nal_emittere(m->sps, rbsp, bs_longitudo(&bs));
}

static void pps_genera(pfr_mp4_t *m)
{
    uint8_t rbsp[64];
    bs_t bs;
    bs_initia(&bs, rbsp, sizeof(rbsp));

    /* NAL header */
    bs_u(&bs, 1, 0);       /* forbidden_zero_bit */
    bs_u(&bs, 2, 3);       /* nal_ref_idc = 3 */
    bs_u(&bs, 5, 8);       /* nal_unit_type = 8 (PPS) */

    /* PPS */
    bs_ue(&bs, 0);         /* pic_parameter_set_id */
    bs_ue(&bs, 0);         /* seq_parameter_set_id */
    bs_u(&bs, 1, 0);       /* entropy_coding_mode_flag = 0 (CAVLC) */
    bs_u(&bs, 1, 0);       /* bottom_field_pic_order_in_frame_present */
    bs_ue(&bs, 0);         /* num_slice_groups_minus1 */
    bs_ue(&bs, 0);         /* num_ref_idx_l0_default_active_minus1 */
    bs_ue(&bs, 0);         /* num_ref_idx_l1_default_active_minus1 */
    bs_u(&bs, 1, 0);       /* weighted_pred_flag */
    bs_u(&bs, 2, 0);       /* weighted_bipred_idc */
    bs_se(&bs, 0);         /* pic_init_qp_minus26 */
    bs_se(&bs, 0);         /* pic_init_qs_minus26 */
    bs_se(&bs, 0);         /* chroma_qp_index_offset */
    bs_u(&bs, 1, 0);       /* deblocking_filter_control_present */
    bs_u(&bs, 1, 0);       /* constrained_intra_pred_flag */
    bs_u(&bs, 1, 0);       /* redundant_pic_cnt_present */
    bs_rbsp_fini(&bs);

    m->pps_lon = nal_emittere(m->pps, rbsp, bs_longitudo(&bs));
}

/* ================================================================
 * tabulam H.264 I_PCM encodare
 * ================================================================ */

static int tabulam_encoda(pfr_mp4_t *m, const uint32_t *pixels)
{
    argb_ad_yuv420(
        pixels, m->latitudo, m->altitudo,
        m->y_plan, m->cb_plan, m->cr_plan
    );

    /* RBSP: slice header + MB data */
    int max_rbsp  = 64 + m->mbs_lat * m->mbs_alt * 400;
    uint8_t *rbsp = (uint8_t *)malloc((size_t)max_rbsp);
    if (!rbsp)
        return -1;

    bs_t bs;
    bs_initia(&bs, rbsp, max_rbsp);

    /* NAL header */
    bs_u(&bs, 1, 0);       /* forbidden_zero_bit */
    bs_u(&bs, 2, 3);       /* nal_ref_idc = 3 */
    bs_u(&bs, 5, 5);       /* nal_unit_type = 5 (IDR) */

    /* Slice header */
    bs_ue(&bs, 0);         /* first_mb_in_slice = 0 */
    bs_ue(&bs, 7);         /* slice_type = 7 (I, omnia) */
    bs_ue(&bs, 0);         /* pic_parameter_set_id */
    bs_u(&bs, 4, (uint32_t)(m->numerus % 16));  /* frame_num (log2_max=4) */
    /* no field_pic_flag (frame_mbs_only=1) */
    /* no pic_order_cnt (type=2) */
    bs_ue(&bs, (uint32_t)(m->numerus % 2));  /* idr_pic_id */

    /* dec_ref_pic_marking */
    bs_u(&bs, 1, 0);       /* no_output_of_prior_pics_flag */
    bs_u(&bs, 1, 0);       /* long_term_reference_flag */

    bs_se(&bs, 0);         /* slice_qp_delta */
    /* no deblocking (not present) */

    /* Macroblocks — I_PCM */
    for (int mb_y = 0; mb_y < m->mbs_alt; mb_y++) {
        for (int mb_x = 0; mb_x < m->mbs_lat; mb_x++) {
            /* mb_type = 25 (I_PCM) as ue(v) */
            bs_ue(&bs, 25);

            /* byte align */
            while (bs.bit_pos != 0)
                bs_u(&bs, 1, 0);

            /* Y samples: 16x16 */
            for (int y = 0; y < 16; y++) {
                for (int x = 0; x < 16; x++) {
                    int sy      = mb_y * 16 + y;
                    int sx      = mb_x * 16 + x;
                    uint8_t val = 16; /* default */
                    if (sy < m->altitudo && sx < m->latitudo)
                        val = m->y_plan[sy * m->latitudo + sx];
                    rbsp[bs.byte_pos++] = val;
                }
            }

            /* Cb samples: 8x8 */
            int lat_c = m->latitudo / 2;
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int sy      = mb_y * 8 + y;
                    int sx      = mb_x * 8 + x;
                    uint8_t val = 128;
                    if (sy < m->altitudo / 2 && sx < lat_c)
                        val = m->cb_plan[sy * lat_c + sx];
                    rbsp[bs.byte_pos++] = val;
                }
            }

            /* Cr samples: 8x8 */
            for (int y = 0; y < 8; y++) {
                for (int x = 0; x < 8; x++) {
                    int sy      = mb_y * 8 + y;
                    int sx      = mb_x * 8 + x;
                    uint8_t val = 128;
                    if (sy < m->altitudo / 2 && sx < lat_c)
                        val = m->cr_plan[sy * lat_c + sx];
                    rbsp[bs.byte_pos++] = val;
                }
            }
        }
    }

    /* RBSP trailing bits */
    bs.bit_pos = 0; /* iam byte-aligned post I_PCM */
    bs_rbsp_fini(&bs);

    int rbsp_lon = bs.byte_pos;

    /* emulation prevention */
    if (m->nal_cap < rbsp_lon * 2) {
        m->nal_cap = rbsp_lon * 2;
        m->nal_alveus = (uint8_t *)realloc(
            m->nal_alveus,
            (size_t)m->nal_cap
        );
    }
    int nal_lon = nal_emittere(m->nal_alveus, rbsp, rbsp_lon);
    free(rbsp);

    /* scribe in mdat: 4-byte length + NAL data */
    mp4_u32(m->plica, (uint32_t)nal_lon);
    fwrite(m->nal_alveus, 1, (size_t)nal_lon, m->plica);

    /* magnitudinem serva */
    uint32_t sample_mag = 4 + (uint32_t)nal_lon;
    if (m->numerus >= m->capacitas) {
        m->capacitas = m->capacitas ? m->capacitas * 2 : 256;
        m->magnitudines = (uint32_t *)realloc(
            m->magnitudines,
            (size_t)m->capacitas * sizeof(uint32_t)
        );
    }
    m->magnitudines[m->numerus] = sample_mag;

    return 0;
}

/* ================================================================
 * MP4 capsas scribere (moov)
 * ================================================================ */

static void moov_scribe(pfr_mp4_t *m)
{
    FILE *f       = m->plica;
    int n         = m->numerus;
    int dur_media = n;                     /* in media timescale (= fps) */
    int dur_movie = n * 1000 / m->fps;     /* in movie timescale (= 1000) */
    long pos_moov, pos_trak, pos_mdia, pos_minf, pos_stbl;
    long pos_stsd, pos_avc1;

    mp4_capsam_initia(f, "moov", &pos_moov);

    /* --- mvhd --- */
    {
        long p;
        mp4_capsam_initia(f, "mvhd", &p);
        mp4_u32(f, 0);             /* version + flags */
        mp4_u32(f, 0);             /* creation_time */
        mp4_u32(f, 0);             /* modification_time */
        mp4_u32(f, 1000);          /* timescale */
        mp4_u32(f, (uint32_t)dur_movie);  /* duration */
        mp4_u32(f, 0x00010000);    /* rate = 1.0 */
        mp4_u16(f, 0x0100);        /* volume = 1.0 */
        mp4_zeros(f, 10);          /* reserved */
        /* matrix (identity) */
        mp4_u32(f, 0x00010000);
        mp4_zeros(f, 4);
        mp4_zeros(f, 4);
        mp4_zeros(f, 4);
        mp4_u32(f, 0x00010000);
        mp4_zeros(f, 4);
        mp4_zeros(f, 4);
        mp4_zeros(f, 4);
        mp4_u32(f, 0x40000000);
        mp4_zeros(f, 24);          /* pre_defined */
        mp4_u32(f, 2);             /* next_track_ID */
        mp4_capsam_fini(f, p);
    }

    mp4_capsam_initia(f, "trak", &pos_trak);

    /* --- tkhd --- */
    {
        long p;
        mp4_capsam_initia(f, "tkhd", &p);
        mp4_u32(f, 0x00000003);    /* version=0, flags=3 (enabled+in_movie) */
        mp4_u32(f, 0);             /* creation */
        mp4_u32(f, 0);             /* modification */
        mp4_u32(f, 1);             /* track_ID */
        mp4_u32(f, 0);             /* reserved */
        mp4_u32(f, (uint32_t)dur_movie);
        mp4_zeros(f, 8);           /* reserved */
        mp4_u16(f, 0);             /* layer */
        mp4_u16(f, 0);             /* alternate_group */
        mp4_u16(f, 0);             /* volume */
        mp4_u16(f, 0);             /* reserved */
        /* matrix */
        mp4_u32(f, 0x00010000);
        mp4_zeros(f, 4);
        mp4_zeros(f, 4);
        mp4_zeros(f, 4);
        mp4_u32(f, 0x00010000);
        mp4_zeros(f, 4);
        mp4_zeros(f, 4);
        mp4_zeros(f, 4);
        mp4_u32(f, 0x40000000);
        mp4_u32(f, (uint32_t)m->latitudo << 16);   /* width (fixed-point) */
        mp4_u32(f, (uint32_t)m->altitudo << 16);   /* height */
        mp4_capsam_fini(f, p);
    }

    mp4_capsam_initia(f, "mdia", &pos_mdia);

    /* --- mdhd --- */
    {
        long p;
        mp4_capsam_initia(f, "mdhd", &p);
        mp4_u32(f, 0);
        mp4_u32(f, 0);
        mp4_u32(f, 0);
        mp4_u32(f, (uint32_t)m->fps);      /* timescale */
        mp4_u32(f, (uint32_t)dur_media);    /* duration */
        mp4_u16(f, 0x55C4);        /* language = und */
        mp4_u16(f, 0);
        mp4_capsam_fini(f, p);
    }

    /* --- hdlr --- */
    {
        long p;
        mp4_capsam_initia(f, "hdlr", &p);
        mp4_u32(f, 0);
        mp4_u32(f, 0);             /* pre_defined */
        fwrite("vide", 1, 4, f);   /* handler_type */
        mp4_zeros(f, 12);          /* reserved */
        fwrite("VideoHandler", 1, 13, f); /* name (null-terminated) */
        mp4_capsam_fini(f, p);
    }

    mp4_capsam_initia(f, "minf", &pos_minf);

    /* --- vmhd --- */
    {
        long p;
        mp4_capsam_initia(f, "vmhd", &p);
        mp4_u32(f, 0x00000001);    /* version=0, flags=1 */
        mp4_u16(f, 0);
        mp4_u16(f, 0);
        mp4_u16(f, 0);
        mp4_u16(f, 0);
        mp4_capsam_fini(f, p);
    }

    /* --- dinf/dref --- */
    {
        long pd;
        mp4_capsam_initia(f, "dinf", &pd);
        long pr;
        mp4_capsam_initia(f, "dref", &pr);
        mp4_u32(f, 0);             /* version + flags */
        mp4_u32(f, 1);             /* entry_count */
        /* url entry (self-contained) */
        mp4_u32(f, 12);            /* size */
        fwrite("url ", 1, 4, f);
        mp4_u32(f, 0x00000001);    /* flags = self-contained */
        mp4_capsam_fini(f, pr);
        mp4_capsam_fini(f, pd);
    }

    mp4_capsam_initia(f, "stbl", &pos_stbl);

    /* --- stsd (avc1 + avcC) --- */
    mp4_capsam_initia(f, "stsd", &pos_stsd);
    mp4_u32(f, 0);                 /* version + flags */
    mp4_u32(f, 1);                 /* entry_count */

    mp4_capsam_initia(f, "avc1", &pos_avc1);
    mp4_zeros(f, 6);               /* reserved */
    mp4_u16(f, 1);                 /* data_reference_index */
    mp4_zeros(f, 16);              /* pre_defined + reserved */
    mp4_u16(f, (uint16_t)m->latitudo);
    mp4_u16(f, (uint16_t)m->altitudo);
    mp4_u32(f, 0x00480000);        /* horiz resolution 72 dpi */
    mp4_u32(f, 0x00480000);        /* vert resolution 72 dpi */
    mp4_u32(f, 0);                 /* reserved */
    mp4_u16(f, 1);                 /* frame_count */
    mp4_zeros(f, 32);              /* compressorname */
    mp4_u16(f, 0x0018);            /* depth = 24 */
    mp4_u16(f, 0xFFFF);            /* pre_defined = -1 */

    /* avcC box */
    {
        long pa;
        mp4_capsam_initia(f, "avcC", &pa);
        mp4_u8(f, 1);              /* version */
        mp4_u8(f, 66);             /* profile */
        mp4_u8(f, 0xC0);           /* compatibility (constraint_set0+1) */
        mp4_u8(f, 30);             /* level */
        mp4_u8(f, 0xFF);           /* 6 reserved bits + lengthSizeMinusOne=3 */
        mp4_u8(f, 0xE1);           /* 3 reserved bits + numSPS=1 */
        mp4_u16(f, (uint16_t)m->sps_lon);
        fwrite(m->sps, 1, (size_t)m->sps_lon, f);
        mp4_u8(f, 1);              /* numPPS */
        mp4_u16(f, (uint16_t)m->pps_lon);
        fwrite(m->pps, 1, (size_t)m->pps_lon, f);
        mp4_capsam_fini(f, pa);
    }

    mp4_capsam_fini(f, pos_avc1);
    mp4_capsam_fini(f, pos_stsd);

    /* --- stts --- */
    {
        long p;
        mp4_capsam_initia(f, "stts", &p);
        mp4_u32(f, 0);
        mp4_u32(f, 1);             /* entry_count */
        mp4_u32(f, (uint32_t)n);   /* sample_count */
        mp4_u32(f, 1);             /* sample_delta (1 tick per frame) */
        mp4_capsam_fini(f, p);
    }

    /* --- stsc --- */
    {
        long p;
        mp4_capsam_initia(f, "stsc", &p);
        mp4_u32(f, 0);
        mp4_u32(f, 1);             /* entry_count */
        mp4_u32(f, 1);             /* first_chunk */
        mp4_u32(f, (uint32_t)n);   /* samples_per_chunk (omnia in uno chunk) */
        mp4_u32(f, 1);             /* sample_description_index */
        mp4_capsam_fini(f, p);
    }

    /* --- stsz --- */
    {
        long p;
        mp4_capsam_initia(f, "stsz", &p);
        mp4_u32(f, 0);
        mp4_u32(f, 0);             /* sample_size = 0 (variabilis) */
        mp4_u32(f, (uint32_t)n);
        for (int i = 0; i < n; i++)
            mp4_u32(f, m->magnitudines[i]);
        mp4_capsam_fini(f, p);
    }

    /* --- stco --- */
    {
        long p;
        mp4_capsam_initia(f, "stco", &p);
        mp4_u32(f, 0);
        mp4_u32(f, 1);             /* entry_count (unus chunk) */
        mp4_u32(f, (uint32_t)m->mdat_data_pos);  /* chunk offset */
        mp4_capsam_fini(f, p);
    }

    /* --- stss (sync samples — omnes sunt sync) --- */
    {
        long p;
        mp4_capsam_initia(f, "stss", &p);
        mp4_u32(f, 0);
        mp4_u32(f, (uint32_t)n);
        for (int i = 0; i < n; i++)
            mp4_u32(f, (uint32_t)(i + 1));  /* 1-indexed */
        mp4_capsam_fini(f, p);
    }

    mp4_capsam_fini(f, pos_stbl);
    mp4_capsam_fini(f, pos_minf);
    mp4_capsam_fini(f, pos_mdia);
    mp4_capsam_fini(f, pos_trak);
    mp4_capsam_fini(f, pos_moov);
}

/* ================================================================
 * interfacies publica
 * ================================================================ */

pfr_mp4_t *pfr_mp4_initia(const char *via, int lat, int alt, int fps)
{
    pfr_mp4_t *m = (pfr_mp4_t *)calloc(1, sizeof(*m));
    if (!m)
        return NULL;

    m->latitudo = lat;
    m->altitudo = alt;
    m->fps      = fps;
    m->mbs_lat  = (lat + 15) / 16;
    m->mbs_alt  = (alt + 15) / 16;

    m->plica = fopen(via, "wb");
    if (!m->plica) {
        free(m);
        return NULL;
    }

    /* genera SPS et PPS */
    sps_genera(m);
    pps_genera(m);

    /* plana YUV */
    int n_y    = (m->mbs_lat * 16) * (m->mbs_alt * 16);
    int n_c    = (m->mbs_lat * 8)  * (m->mbs_alt * 8);
    m->y_plan  = (uint8_t *)calloc(1, (size_t)n_y);
    m->cb_plan = (uint8_t *)calloc(1, (size_t)n_c);
    m->cr_plan = (uint8_t *)calloc(1, (size_t)n_c);

    /* ftyp */
    {
        long p;
        mp4_capsam_initia(m->plica, "ftyp", &p);
        fwrite("isom", 1, 4, m->plica);    /* major_brand */
        mp4_u32(m->plica, 0x200);           /* minor_version */
        fwrite("isom", 1, 4, m->plica);
        fwrite("iso2", 1, 4, m->plica);
        fwrite("avc1", 1, 4, m->plica);
        fwrite("mp41", 1, 4, m->plica);
        mp4_capsam_fini(m->plica, p);
    }

    /* mdat — magnitudinem postea replemus */
    m->mdat_pos = ftell(m->plica);
    mp4_u32(m->plica, 0);          /* placeholder */
    fwrite("mdat", 1, 4, m->plica);
    m->mdat_data_pos = ftell(m->plica);

    return m;
}

int pfr_mp4_tabulam_adde(pfr_mp4_t *m, const uint32_t *pixels)
{
    if (!m || !pixels)
        return -1;
    int res = tabulam_encoda(m, pixels);
    if (res == 0)
        m->numerus++;
    return res;
}

void pfr_mp4_fini(pfr_mp4_t *m)
{
    if (!m)
        return;

    if (m->plica && m->numerus > 0) {
        /* mdat magnitudinem reple */
        long mdat_finis   = ftell(m->plica);
        uint32_t mdat_mag = (uint32_t)(mdat_finis - m->mdat_pos);
        fseek(m->plica, m->mdat_pos, SEEK_SET);
        mp4_u32(m->plica, mdat_mag);
        fseek(m->plica, mdat_finis, SEEK_SET);

        /* moov scribe */
        moov_scribe(m);
    }

    if (m->plica)
        fclose(m->plica);
    free(m->y_plan);
    free(m->cb_plan);
    free(m->cr_plan);
    free(m->magnitudines);
    free(m->nal_alveus);
    free(m);
}
