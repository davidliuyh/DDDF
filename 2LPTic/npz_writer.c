#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <math.h>
#include <sys/types.h>

#include <gsl/gsl_rng.h>

#include "proto.h"

#define NPZ_PI 3.14159265358979323846

struct npz_entry
{
  const char *name;
  uint64_t size;
  uint32_t crc32;
  uint64_t local_header_offset;
  int use_zip64;
};

static uint32_t crc32_init(void)
{
  return 0xFFFFFFFFu;
}

static uint32_t crc32_update(uint32_t crc, const unsigned char *buf, size_t len)
{
  size_t i;
  int j;

  for(i = 0; i < len; i++)
    {
      crc ^= (uint32_t) buf[i];
      for(j = 0; j < 8; j++)
        crc = (crc >> 1) ^ (0xEDB88320u & (-(int32_t) (crc & 1u)));
    }

  return crc;
}

static uint32_t crc32_finalize(uint32_t crc)
{
  return crc ^ 0xFFFFFFFFu;
}

static uint32_t crc32_bytes(const unsigned char *buf, size_t len)
{
  return crc32_finalize(crc32_update(crc32_init(), buf, len));
}

static int append_bytes(unsigned char **buf, uint32_t *len, uint32_t *cap, const void *src, uint32_t src_len)
{
  unsigned char *new_buf;

  if(*len + src_len > *cap)
    {
      uint32_t new_cap = (*cap == 0) ? 256 : *cap;
      while(new_cap < *len + src_len)
        new_cap *= 2;
      new_buf = (unsigned char *) realloc(*buf, new_cap);
      if(new_buf == NULL)
        return -1;
      *buf = new_buf;
      *cap = new_cap;
    }

  memcpy((*buf) + *len, src, src_len);
  *len += src_len;
  return 0;
}

static int build_npy_payload(const char *descr,
                             const void *array_data,
                             uint32_t elem_count,
                             uint32_t elem_size,
                             int ndims,
                             const uint32_t *shape,
                             unsigned char **out_buf,
                             uint32_t *out_len)
{
  unsigned char *buf = NULL;
  uint32_t len = 0, cap = 0;
  char header[512];
  char shape_text[128];
  int n;
  uint16_t header_len;
  uint32_t preamble_len;
  uint32_t pad;
  unsigned char magic[8] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0};

  if(ndims == 1)
    snprintf(shape_text, sizeof(shape_text), "(%u,)", shape[0]);
  else if(ndims == 2)
    snprintf(shape_text, sizeof(shape_text), "(%u, %u)", shape[0], shape[1]);
  else
    return -1;

  n = snprintf(header,
               sizeof(header),
               "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
               descr,
               shape_text);
  if(n <= 0 || n >= (int) sizeof(header) - 2)
    return -1;

  preamble_len = 10;
  pad = 16 - ((preamble_len + (uint32_t) n + 1) % 16);
  if(pad == 16)
    pad = 0;

  while(pad > 0)
    {
      header[n++] = ' ';
      pad--;
    }
  header[n++] = '\n';

  if(n > 65535)
    return -1;

  header_len = (uint16_t) n;

  if(append_bytes(&buf, &len, &cap, magic, sizeof(magic)) != 0)
    goto fail;
  if(append_bytes(&buf, &len, &cap, &header_len, sizeof(header_len)) != 0)
    goto fail;
  if(append_bytes(&buf, &len, &cap, header, header_len) != 0)
    goto fail;
  if(append_bytes(&buf, &len, &cap, array_data, elem_count * elem_size) != 0)
    goto fail;

  *out_buf = buf;
  *out_len = len;
  return 0;

fail:
  free(buf);
  return -1;
}

static int build_npy_header(const char *descr,
                            int ndims,
                            const uint64_t *shape,
                            unsigned char *header,
                            uint16_t *header_len)
{
  char dict[512];
  char shape_text[256];
  int n;
  uint32_t preamble_len;
  uint32_t pad;

  if(ndims == 1)
    snprintf(shape_text, sizeof(shape_text), "(%" PRIu64 ",)", shape[0]);
  else if(ndims == 2)
    snprintf(shape_text, sizeof(shape_text), "(%" PRIu64 ", %" PRIu64 ")", shape[0], shape[1]);
  else if(ndims == 3)
    snprintf(shape_text,
             sizeof(shape_text),
             "(%" PRIu64 ", %" PRIu64 ", %" PRIu64 ")",
             shape[0],
             shape[1],
             shape[2]);
  else
    return -1;

  n = snprintf(dict,
               sizeof(dict),
               "{'descr': '%s', 'fortran_order': False, 'shape': %s, }",
               descr,
               shape_text);
  if(n <= 0 || n >= (int) sizeof(dict) - 2)
    return -1;

  preamble_len = 10;
  pad = 16 - ((preamble_len + (uint32_t) n + 1) % 16);
  if(pad == 16)
    pad = 0;

  while(pad > 0)
    dict[n++] = ' ', pad--;
  dict[n++] = '\n';

  if(n > 65535)
    return -1;

  memcpy(header, dict, (size_t) n);
  *header_len = (uint16_t) n;
  return 0;
}

static int write_le16(FILE *fd, uint16_t v)
{
  return fwrite(&v, sizeof(v), 1, fd) == 1 ? 0 : -1;
}

static int write_le32(FILE *fd, uint32_t v)
{
  return fwrite(&v, sizeof(v), 1, fd) == 1 ? 0 : -1;
}

static int write_le64(FILE *fd, uint64_t v)
{
  return fwrite(&v, sizeof(v), 1, fd) == 1 ? 0 : -1;
}

static int file_seek64(FILE *fd, uint64_t pos)
{
  return fseeko(fd, (off_t) pos, SEEK_SET);
}

static uint64_t file_tell64(FILE *fd)
{
  off_t p = ftello(fd);
  if(p < 0)
    return UINT64_MAX;
  return (uint64_t) p;
}

static int write_local_header(FILE *fd,
                              const char *name,
                              uint64_t size,
                              uint32_t crc32,
                              int use_zip64,
                              uint64_t *local_header_offset,
                              uint64_t *crc_field_offset)
{
  uint16_t name_len = (uint16_t) strlen(name);
  uint16_t extra_len = use_zip64 ? 20 : 0;
  uint64_t offset = file_tell64(fd);

  if(offset == UINT64_MAX)
    return -1;

  if(write_le32(fd, 0x04034b50u) != 0)
    return -1;
  if(write_le16(fd, (uint16_t) (use_zip64 ? 45 : 20)) != 0)
    return -1;
  if(write_le16(fd, 0) != 0)
    return -1;
  if(write_le16(fd, 0) != 0)
    return -1;
  if(write_le16(fd, 0) != 0)
    return -1;
  if(write_le16(fd, 0) != 0)
    return -1;

  if(crc_field_offset)
    *crc_field_offset = offset + 14;

  if(write_le32(fd, crc32) != 0)
    return -1;
  if(write_le32(fd, use_zip64 ? 0xFFFFFFFFu : (uint32_t) size) != 0)
    return -1;
  if(write_le32(fd, use_zip64 ? 0xFFFFFFFFu : (uint32_t) size) != 0)
    return -1;
  if(write_le16(fd, name_len) != 0)
    return -1;
  if(write_le16(fd, extra_len) != 0)
    return -1;
  if(fwrite(name, 1, name_len, fd) != name_len)
    return -1;

  if(use_zip64)
    {
      if(write_le16(fd, 0x0001) != 0)
        return -1;
      if(write_le16(fd, 16) != 0)
        return -1;
      if(write_le64(fd, size) != 0)
        return -1;
      if(write_le64(fd, size) != 0)
        return -1;
    }

  if(local_header_offset)
    *local_header_offset = offset;

  return 0;
}

static int patch_crc32(FILE *fd, uint64_t crc_field_offset, uint32_t crc32)
{
  uint64_t back = file_tell64(fd);

  if(back == UINT64_MAX)
    return -1;
  if(file_seek64(fd, crc_field_offset) != 0)
    return -1;
  if(write_le32(fd, crc32) != 0)
    return -1;
  if(file_seek64(fd, back) != 0)
    return -1;

  return 0;
}

static int write_small_entry(FILE *fd,
                             struct npz_entry *entry,
                             const char *name,
                             const unsigned char *payload,
                             uint64_t payload_size)
{
  uint32_t crc = crc32_bytes(payload, (size_t) payload_size);

  if(write_local_header(fd,
                        name,
                        payload_size,
                        crc,
                        payload_size > 0xFFFFFFFFull,
                        &entry->local_header_offset,
                        NULL) != 0)
    return -1;

  if(fwrite(payload, 1, (size_t) payload_size, fd) != (size_t) payload_size)
    return -1;

  entry->name = name;
  entry->size = payload_size;
  entry->crc32 = crc;
  entry->use_zip64 = (payload_size > 0xFFFFFFFFull);

  return 0;
}

static int add_scalar_i32_entry(FILE *fd, struct npz_entry *entry, const char *name, int32_t value)
{
  unsigned char *payload = NULL;
  uint32_t payload_size = 0;
  uint32_t shape1[1] = {1};
  int rc;

  if(build_npy_payload("<i4", &value, 1, sizeof(value), 1, shape1, &payload, &payload_size) != 0)
    return -1;

  rc = write_small_entry(fd, entry, name, payload, payload_size);
  free(payload);
  return rc;
}

static int add_scalar_f64_entry(FILE *fd, struct npz_entry *entry, const char *name, double value)
{
  unsigned char *payload = NULL;
  uint32_t payload_size = 0;
  uint32_t shape1[1] = {1};
  int rc;

  if(build_npy_payload("<f8", &value, 1, sizeof(value), 1, shape1, &payload, &payload_size) != 0)
    return -1;

  rc = write_small_entry(fd, entry, name, payload, payload_size);
  free(payload);
  return rc;
}

static int write_npy_preamble(FILE *fd,
                              const unsigned char *header,
                              uint16_t header_len,
                              uint32_t *crc_state)
{
  unsigned char preamble[10] = {0x93, 'N', 'U', 'M', 'P', 'Y', 1, 0, 0, 0};

  preamble[8] = (unsigned char) (header_len & 0xFFu);
  preamble[9] = (unsigned char) ((header_len >> 8) & 0xFFu);

  if(fwrite(preamble, 1, sizeof(preamble), fd) != sizeof(preamble))
    return -1;
  if(fwrite(header, 1, header_len, fd) != header_len)
    return -1;

  *crc_state = crc32_update(*crc_state, preamble, sizeof(preamble));
  *crc_state = crc32_update(*crc_state, header, header_len);

  return 0;
}

static void fill_white_noise_row(float *row,
                                 int i,
                                 int j,
                                 int nmesh,
                                 const unsigned int *seedtable,
                                 int rayleigh_sampling,
                                 double phase_shift,
                                 gsl_rng *rng_primary,
                                 gsl_rng *rng_partner)
{
  int k;
  int nk = nmesh / 2 + 1;
  int partner_i = 0;
  int partner_j = 0;
  int use_conjugate = 0;
  double phase;
  double ampl_u;
  double amp_factor;

  memset(row, 0, sizeof(float) * (size_t) (2 * nk));

  gsl_rng_set(rng_primary, seedtable[(size_t) i * (size_t) nmesh + (size_t) j]);

  for(k = 0; k < nmesh / 2; k++)
    {
      phase = gsl_rng_uniform(rng_primary) * 2.0 * NPZ_PI + phase_shift;

      do
        ampl_u = gsl_rng_uniform(rng_primary);
      while(ampl_u == 0.0);

      if(i == nmesh / 2 || j == nmesh / 2)
        continue;

      if(i == 0 && j == 0 && k == 0)
        continue;

      amp_factor = rayleigh_sampling ? sqrt(-log(ampl_u)) : 1.0;

      if(k > 0)
        {
          row[2 * k] = (float) (amp_factor * cos(phase));
          row[2 * k + 1] = (float) (amp_factor * sin(phase));
        }
      else
        {
          if(i == 0)
            {
              if(j > 0 && j < nmesh / 2)
                {
                  row[0] = (float) (amp_factor * cos(phase));
                  row[1] = (float) (amp_factor * sin(phase));
                }
            }
          else if(i < nmesh / 2)
            {
              row[0] = (float) (amp_factor * cos(phase));
              row[1] = (float) (amp_factor * sin(phase));
            }
        }
    }

  if(i == nmesh / 2 || j == nmesh / 2)
    return;

  if(i == 0)
    {
      if(j > nmesh / 2)
        {
          partner_i = 0;
          partner_j = nmesh - j;
          use_conjugate = 1;
        }
    }
  else if(i > nmesh / 2)
    {
      partner_i = nmesh - i;
      if(partner_i == nmesh)
        partner_i = 0;
      partner_j = nmesh - j;
      if(partner_j == nmesh)
        partner_j = 0;
      use_conjugate = 1;
    }

  if(!use_conjugate)
    return;

  gsl_rng_set(rng_partner, seedtable[(size_t) partner_i * (size_t) nmesh + (size_t) partner_j]);

  phase = gsl_rng_uniform(rng_partner) * 2.0 * NPZ_PI + phase_shift;

  do
    ampl_u = gsl_rng_uniform(rng_partner);
  while(ampl_u == 0.0);

  if(partner_i == nmesh / 2 || partner_j == nmesh / 2 || (partner_i == 0 && partner_j == 0))
    return;

  amp_factor = rayleigh_sampling ? sqrt(-log(ampl_u)) : 1.0;
  row[0] = (float) (amp_factor * cos(phase));
  row[1] = (float) (-amp_factor * sin(phase));
}

static int stream_white_noise_payload(FILE *fd,
                                      const unsigned int *seedtable,
                                      int nmesh,
                                      int rayleigh_sampling,
                                      int phase_flip,
                                      uint64_t *payload_size_out,
                                      uint32_t *crc32_out)
{
  uint64_t shape[3];
  unsigned char header[512];
  uint16_t header_len = 0;
  uint64_t data_bytes;
  int nk = nmesh / 2 + 1;
  float *row = NULL;
  uint32_t crc_state = crc32_init();
  gsl_rng *rng_primary = NULL;
  gsl_rng *rng_partner = NULL;
  int i, j;
  double phase_shift = (phase_flip == 1) ? NPZ_PI : 0.0;

  shape[0] = (uint64_t) nmesh;
  shape[1] = (uint64_t) nmesh;
  shape[2] = (uint64_t) nk;

  if(build_npy_header("<c8", 3, shape, header, &header_len) != 0)
    return -1;

  data_bytes = shape[0] * shape[1] * shape[2] * sizeof(float) * 2ull;
  *payload_size_out = 10ull + (uint64_t) header_len + data_bytes;

  if(write_npy_preamble(fd, header, header_len, &crc_state) != 0)
    return -1;

  row = (float *) malloc(sizeof(float) * (size_t) (2 * nk));
  if(row == NULL)
    goto fail;

  rng_primary = gsl_rng_alloc(gsl_rng_ranlxd1);
  rng_partner = gsl_rng_alloc(gsl_rng_ranlxd1);
  if(rng_primary == NULL || rng_partner == NULL)
    goto fail;

  for(i = 0; i < nmesh; i++)
    {
      for(j = 0; j < nmesh; j++)
        {
          fill_white_noise_row(row,
                               i,
                               j,
                               nmesh,
                               seedtable,
                               rayleigh_sampling,
                               phase_shift,
                               rng_primary,
                               rng_partner);

          if(fwrite(row, sizeof(float), (size_t) (2 * nk), fd) != (size_t) (2 * nk))
            goto fail;

          crc_state = crc32_update(crc_state,
                                   (const unsigned char *) row,
                                   sizeof(float) * (size_t) (2 * nk));
        }

      if((i % 32) == 0)
        {
          printf("white noise export progress: i=%d/%d\n", i, nmesh - 1);
          fflush(stdout);
        }
    }

  *crc32_out = crc32_finalize(crc_state);

  gsl_rng_free(rng_primary);
  gsl_rng_free(rng_partner);
  free(row);
  return 0;

fail:
  if(rng_primary)
    gsl_rng_free(rng_primary);
  if(rng_partner)
    gsl_rng_free(rng_partner);
  free(row);
  return -1;
}

int write_white_noise_npz(const char *path,
                          const unsigned int *seedtable,
                          int nmesh,
                          int nsample,
                          int seed,
                          int rayleigh_sampling,
                          int phase_flip,
                          double box,
                          double dplus,
                          double init_time)
{
  FILE *fd = NULL;
  struct npz_entry entries[16];
  int entry_count = 0;
  uint64_t white_noise_size = 0;
  uint32_t white_noise_crc = 0;
  uint64_t local_offset = 0;
  uint64_t crc_offset = 0;
  int i;
  uint64_t central_dir_start;
  uint64_t central_dir_size;
  int need_zip64_archive = 0;

  fd = fopen(path, "wb");
  if(fd == NULL)
    return -1;

  {
    uint64_t shape[3];
    unsigned char header[512];
    uint16_t header_len = 0;
    int nk = nmesh / 2 + 1;

    shape[0] = (uint64_t) nmesh;
    shape[1] = (uint64_t) nmesh;
    shape[2] = (uint64_t) nk;
    if(build_npy_header("<c8", 3, shape, header, &header_len) != 0)
      goto fail;

    white_noise_size = 10ull + (uint64_t) header_len +
                       shape[0] * shape[1] * shape[2] * sizeof(float) * 2ull;
  }

  if(write_local_header(fd,
                        "white_noise.npy",
                        white_noise_size,
                        0,
                        white_noise_size > 0xFFFFFFFFull,
                        &local_offset,
                        &crc_offset) != 0)
    goto fail;

  if(stream_white_noise_payload(fd,
                                seedtable,
                                nmesh,
                                rayleigh_sampling,
                                phase_flip,
                                &white_noise_size,
                                &white_noise_crc) != 0)
    goto fail;

  if(patch_crc32(fd, crc_offset, white_noise_crc) != 0)
    goto fail;

  entries[entry_count].name = "white_noise.npy";
  entries[entry_count].size = white_noise_size;
  entries[entry_count].crc32 = white_noise_crc;
  entries[entry_count].local_header_offset = local_offset;
  entries[entry_count].use_zip64 = (white_noise_size > 0xFFFFFFFFull);
  entry_count++;

  if(add_scalar_i32_entry(fd, &entries[entry_count++], "nmesh.npy", (int32_t) nmesh) != 0)
    goto fail;
  if(add_scalar_i32_entry(fd, &entries[entry_count++], "nsample.npy", (int32_t) nsample) != 0)
    goto fail;
  if(add_scalar_i32_entry(fd, &entries[entry_count++], "seed.npy", (int32_t) seed) != 0)
    goto fail;
  if(add_scalar_i32_entry(fd,
                          &entries[entry_count++],
                          "rayleigh_sampling.npy",
                          (int32_t) rayleigh_sampling) != 0)
    goto fail;
  if(add_scalar_i32_entry(fd, &entries[entry_count++], "phase_flip.npy", (int32_t) phase_flip) != 0)
    goto fail;

  if(add_scalar_f64_entry(fd, &entries[entry_count++], "box.npy", box) != 0)
    goto fail;
  if(add_scalar_f64_entry(fd, &entries[entry_count++], "dplus.npy", dplus) != 0)
    goto fail;
  if(add_scalar_f64_entry(fd, &entries[entry_count++], "init_time.npy", init_time) != 0)
    goto fail;

  central_dir_start = file_tell64(fd);
  if(central_dir_start == UINT64_MAX)
    goto fail;

  for(i = 0; i < entry_count; i++)
    {
      uint16_t name_len = (uint16_t) strlen(entries[i].name);
      int need_zip64_entry = entries[i].use_zip64 || (entries[i].local_header_offset > 0xFFFFFFFFull);

      if(need_zip64_entry)
        need_zip64_archive = 1;

      if(write_le32(fd, 0x02014b50u) != 0)
        goto fail;
      if(write_le16(fd, (uint16_t) (need_zip64_entry ? 45 : 20)) != 0)
        goto fail;
      if(write_le16(fd, (uint16_t) (need_zip64_entry ? 45 : 20)) != 0)
        goto fail;
      if(write_le16(fd, 0) != 0)
        goto fail;
      if(write_le16(fd, 0) != 0)
        goto fail;
      if(write_le16(fd, 0) != 0)
        goto fail;
      if(write_le16(fd, 0) != 0)
        goto fail;
      if(write_le32(fd, entries[i].crc32) != 0)
        goto fail;
      if(write_le32(fd, need_zip64_entry ? 0xFFFFFFFFu : (uint32_t) entries[i].size) != 0)
        goto fail;
      if(write_le32(fd, need_zip64_entry ? 0xFFFFFFFFu : (uint32_t) entries[i].size) != 0)
        goto fail;
      if(write_le16(fd, name_len) != 0)
        goto fail;
      if(write_le16(fd, (uint16_t) (need_zip64_entry ? 28 : 0)) != 0)
        goto fail;
      if(write_le16(fd, 0) != 0)
        goto fail;
      if(write_le16(fd, 0) != 0)
        goto fail;
      if(write_le16(fd, 0) != 0)
        goto fail;
      if(write_le32(fd, 0) != 0)
        goto fail;
      if(write_le32(fd, need_zip64_entry ? 0xFFFFFFFFu : (uint32_t) entries[i].local_header_offset) != 0)
        goto fail;

      if(fwrite(entries[i].name, 1, name_len, fd) != name_len)
        goto fail;

      if(need_zip64_entry)
        {
          if(write_le16(fd, 0x0001) != 0)
            goto fail;
          if(write_le16(fd, 24) != 0)
            goto fail;
          if(write_le64(fd, entries[i].size) != 0)
            goto fail;
          if(write_le64(fd, entries[i].size) != 0)
            goto fail;
          if(write_le64(fd, entries[i].local_header_offset) != 0)
            goto fail;
        }
    }

  {
    uint64_t central_end = file_tell64(fd);
    uint64_t zip64_eocd_offset;

    if(central_end == UINT64_MAX)
      goto fail;
    central_dir_size = central_end - central_dir_start;

    if(central_dir_start > 0xFFFFFFFFull || central_dir_size > 0xFFFFFFFFull || entry_count > 0xFFFF)
      need_zip64_archive = 1;

    if(need_zip64_archive)
      {
        zip64_eocd_offset = file_tell64(fd);
        if(zip64_eocd_offset == UINT64_MAX)
          goto fail;

        if(write_le32(fd, 0x06064b50u) != 0)
          goto fail;
        if(write_le64(fd, 44) != 0)
          goto fail;
        if(write_le16(fd, 45) != 0)
          goto fail;
        if(write_le16(fd, 45) != 0)
          goto fail;
        if(write_le32(fd, 0) != 0)
          goto fail;
        if(write_le32(fd, 0) != 0)
          goto fail;
        if(write_le64(fd, (uint64_t) entry_count) != 0)
          goto fail;
        if(write_le64(fd, (uint64_t) entry_count) != 0)
          goto fail;
        if(write_le64(fd, central_dir_size) != 0)
          goto fail;
        if(write_le64(fd, central_dir_start) != 0)
          goto fail;

        if(write_le32(fd, 0x07064b50u) != 0)
          goto fail;
        if(write_le32(fd, 0) != 0)
          goto fail;
        if(write_le64(fd, zip64_eocd_offset) != 0)
          goto fail;
        if(write_le32(fd, 1) != 0)
          goto fail;
      }

    if(write_le32(fd, 0x06054b50u) != 0)
      goto fail;
    if(write_le16(fd, 0) != 0)
      goto fail;
    if(write_le16(fd, 0) != 0)
      goto fail;
    if(write_le16(fd, need_zip64_archive ? 0xFFFFu : (uint16_t) entry_count) != 0)
      goto fail;
    if(write_le16(fd, need_zip64_archive ? 0xFFFFu : (uint16_t) entry_count) != 0)
      goto fail;
    if(write_le32(fd, need_zip64_archive ? 0xFFFFFFFFu : (uint32_t) central_dir_size) != 0)
      goto fail;
    if(write_le32(fd, need_zip64_archive ? 0xFFFFFFFFu : (uint32_t) central_dir_start) != 0)
      goto fail;
    if(write_le16(fd, 0) != 0)
      goto fail;
  }

  fclose(fd);
  return 0;

fail:
  if(fd)
    fclose(fd);
  return -1;
}

struct white_noise_npz_stream
{
  FILE *fd;
  uint64_t data_size;
  uint64_t data_read;
  int nmesh;
  int nk;
};

static int read_le16(FILE *fd, uint16_t *v)
{
  return fread(v, sizeof(*v), 1, fd) == 1 ? 0 : -1;
}

static int read_le32(FILE *fd, uint32_t *v)
{
  return fread(v, sizeof(*v), 1, fd) == 1 ? 0 : -1;
}

static uint16_t read_u16_from_bytes(const unsigned char *p)
{
  return (uint16_t) p[0] | ((uint16_t) p[1] << 8);
}

static uint32_t read_u32_from_bytes(const unsigned char *p)
{
  return (uint32_t) p[0] |
         ((uint32_t) p[1] << 8) |
         ((uint32_t) p[2] << 16) |
         ((uint32_t) p[3] << 24);
}

static uint64_t read_u64_from_bytes(const unsigned char *p)
{
  return (uint64_t) p[0] |
         ((uint64_t) p[1] << 8) |
         ((uint64_t) p[2] << 16) |
         ((uint64_t) p[3] << 24) |
         ((uint64_t) p[4] << 32) |
         ((uint64_t) p[5] << 40) |
         ((uint64_t) p[6] << 48) |
         ((uint64_t) p[7] << 56);
}

static int parse_npy_shape3(const char *header, uint64_t *d0, uint64_t *d1, uint64_t *d2)
{
  const char *shape = strstr(header, "'shape':");
  int consumed = 0;

  if(shape == NULL)
    return -1;

  shape = strchr(shape, '(');
  if(shape == NULL)
    return -1;

  if(sscanf(shape,
            "(%" SCNu64 ", %" SCNu64 ", %" SCNu64 ")%n",
            d0,
            d1,
            d2,
            &consumed) < 3)
    return -1;

  (void) consumed;
  return 0;
}

int open_white_noise_npz_stream(const char *path, int nmesh, void **handle_out)
{
  FILE *fd = NULL;
  struct white_noise_npz_stream *stream = NULL;
  int nk = nmesh / 2 + 1;

  if(handle_out == NULL)
    return -1;
  *handle_out = NULL;

  fd = fopen(path, "rb");
  if(fd == NULL)
    return -1;

  while(1)
    {
      uint32_t sig = 0;
      uint16_t name_len = 0, extra_len = 0;
      uint16_t method = 0;
      uint32_t comp32 = 0, uncomp32 = 0;
      uint64_t comp_size = 0, uncomp_size = 0;
      char *name = NULL;
      unsigned char *extra = NULL;
      uint64_t payload_start;

      if(read_le32(fd, &sig) != 0)
        break;

      if(sig != 0x04034b50u)
        goto fail;

      if(file_seek64(fd, file_tell64(fd) + 2) != 0) /* version needed */
        goto fail;
      if(file_seek64(fd, file_tell64(fd) + 2) != 0) /* general purpose bits */
        goto fail;

      if(read_le16(fd, &method) != 0)
        goto fail;

      if(file_seek64(fd, file_tell64(fd) + 2) != 0) /* mtime */
        goto fail;
      if(file_seek64(fd, file_tell64(fd) + 2) != 0) /* mdate */
        goto fail;
      if(file_seek64(fd, file_tell64(fd) + 4) != 0) /* crc */
        goto fail;

      if(read_le32(fd, &comp32) != 0)
        goto fail;
      if(read_le32(fd, &uncomp32) != 0)
        goto fail;

      if(read_le16(fd, &name_len) != 0)
        goto fail;
      if(read_le16(fd, &extra_len) != 0)
        goto fail;

      name = (char *) malloc((size_t) name_len + 1);
      if(name == NULL)
        goto fail;
      if(fread(name, 1, name_len, fd) != name_len)
        goto fail;
      name[name_len] = '\0';

      if(extra_len > 0)
        {
          size_t pos = 0;
          extra = (unsigned char *) malloc(extra_len);
          if(extra == NULL)
            goto fail;
          if(fread(extra, 1, extra_len, fd) != extra_len)
            goto fail;

          comp_size = comp32;
          uncomp_size = uncomp32;

          while(pos + 4 <= extra_len)
            {
              uint16_t header_id = read_u16_from_bytes(extra + pos);
              uint16_t data_size = read_u16_from_bytes(extra + pos + 2);
              const unsigned char *data_ptr = extra + pos + 4;

              pos += 4;
              if(pos + data_size > extra_len)
                goto fail;

              if(header_id == 0x0001)
                {
                  size_t z = 0;
                  if(uncomp32 == 0xFFFFFFFFu)
                    {
                      if(z + 8 > data_size)
                        goto fail;
                      uncomp_size = read_u64_from_bytes(data_ptr + z);
                      z += 8;
                    }
                  if(comp32 == 0xFFFFFFFFu)
                    {
                      if(z + 8 > data_size)
                        goto fail;
                      comp_size = read_u64_from_bytes(data_ptr + z);
                      z += 8;
                    }
                }

              pos += data_size;
            }
        }
      else
        {
          comp_size = comp32;
          uncomp_size = uncomp32;
        }

      payload_start = file_tell64(fd);
      if(payload_start == UINT64_MAX)
        goto fail;

      if(method != 0)
        goto fail;

      if(strcmp(name, "white_noise.npy") == 0)
        {
          unsigned char preamble[10];
          unsigned char *header = NULL;
          uint16_t header_len;
          uint64_t d0 = 0, d1 = 0, d2 = 0;
          uint64_t expected_data_size;

          if(uncomp_size < 10)
            goto fail;

          if(fread(preamble, 1, sizeof(preamble), fd) != sizeof(preamble))
            goto fail;

          if(!(preamble[0] == 0x93 && preamble[1] == 'N' && preamble[2] == 'U' && preamble[3] == 'M' &&
               preamble[4] == 'P' && preamble[5] == 'Y'))
            goto fail;

          if(!(preamble[6] == 1 && preamble[7] == 0))
            goto fail;

          header_len = (uint16_t) preamble[8] | ((uint16_t) preamble[9] << 8);
          if(uncomp_size < (uint64_t) (10 + header_len))
            goto fail;

          header = (unsigned char *) malloc((size_t) header_len + 1);
          if(header == NULL)
            goto fail;
          if(fread(header, 1, header_len, fd) != header_len)
            goto fail;
          header[header_len] = '\0';

          if(parse_npy_shape3((const char *) header, &d0, &d1, &d2) != 0)
            goto fail;

          free(header);
          header = NULL;

          if(d0 != (uint64_t) nmesh || d1 != (uint64_t) nmesh || d2 != (uint64_t) nk)
            goto fail;

          expected_data_size = (uint64_t) nmesh * (uint64_t) nmesh * (uint64_t) nk * 2ull * sizeof(float);
          if(uncomp_size != (uint64_t) (10 + header_len) + expected_data_size)
            goto fail;

          stream = (struct white_noise_npz_stream *) malloc(sizeof(*stream));
          if(stream == NULL)
            goto fail;

          stream->fd = fd;
          stream->data_size = expected_data_size;
          stream->data_read = 0;
          stream->nmesh = nmesh;
          stream->nk = nk;
          *handle_out = stream;

          free(name);
          free(extra);
          return 0;
        }

      if(file_seek64(fd, payload_start + comp_size) != 0)
        goto fail;

      free(name);
      free(extra);
      name = NULL;
      extra = NULL;
    }

fail:
  if(fd)
    fclose(fd);
  free(stream);
  return -1;
}

int read_white_noise_npz_row(void *handle, float *row_out, int nk)
{
  struct white_noise_npz_stream *stream = (struct white_noise_npz_stream *) handle;
  uint64_t row_bytes;

  if(stream == NULL || row_out == NULL)
    return -1;
  if(nk != stream->nk)
    return -1;

  row_bytes = (uint64_t) (2 * nk) * sizeof(float);
  if(stream->data_read + row_bytes > stream->data_size)
    return -1;

  if(fread(row_out, 1, (size_t) row_bytes, stream->fd) != (size_t) row_bytes)
    return -1;

  stream->data_read += row_bytes;
  return 0;
}

void close_white_noise_npz_stream(void *handle)
{
  struct white_noise_npz_stream *stream = (struct white_noise_npz_stream *) handle;

  if(stream == NULL)
    return;

  if(stream->fd)
    fclose(stream->fd);
  free(stream);
}
