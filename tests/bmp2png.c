/**
 * BMP2PNG Converter with libpng
 * Copyright (C) 2018 MyungJoo Ham <myungjoo.ham@samsung.com>
 *
 * This library is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Library General Public
 * License as published by the Free Software Foundation; either
 * version 2 of the License, or (at your option) any later version.
 *
 * This library is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Library General Public License for more details.
 *
 */
/**
 * @file	bmp2png.c
 * @date	13 Jul 2018
 * @brief	Simple bmp2png converter for testcases
 * @see		http://github.com/nnsuite/nnstreamer
 * @author	MyungJoo Ham <myungjoo.ham@samsung.com>
 * @bug		No known bugs except for NYI items
 *
 * This converts bmp files created by gen24bBMP.py.
 * This won't support general bmp files.
 *
 * Adopted code from https://www.lemoda.net/c/write-png/
 * The author, "Ben Bullock <benkasminbullock@gmail.com>", has authorized
 * to adopt the code as LGPL-2.1 on 2018-07-13
 */

#include <png.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <glib.h>

typedef enum
{
  RGB = 0,
  GRAY8,
} colorformat_t;

typedef union
{
  struct
  {
    uint8_t red;
    uint8_t green;
    uint8_t blue;
  };
  struct
  {
    uint8_t gray;
  };
} pixel_t;

typedef struct
{
  pixel_t *pixels;
  size_t width;
  size_t height;
  colorformat_t color_format;
} bitmap_t;

/**
 * @brief Given "bitmap", this returns the pixel of bitmap at the point
 *  ("x", "y").
 */
static pixel_t *
pixel_at (bitmap_t * bitmap, int x, int y)
{
  return bitmap->pixels + bitmap->width * y + x;
}

/**
 * @brief Write "bitmap" to a PNG file specified by "path"
 * @return 0 on success, non-zero on error.
 */
static int
save_png_to_file (bitmap_t * bitmap, const char *path)
{
  FILE *fp;
  png_structp png_ptr = NULL;
  png_infop info_ptr = NULL;
  size_t x, y;
  png_byte **row_pointers = NULL;
  /**
   * "status" contains the return value of this function. At first
   * it is set to a value which means 'failure'. When the routine
   * has finished its work, it is set to a value which means
   * 'success'.
   */
  int status = -1;
  /**
   * The following number is set by trial and error only. I cannot
   * see where it it is documented in the libpng manual.
   */
  int pixel_size;
  int depth = 8;
  int color_type;

  if (bitmap->color_format == GRAY8) {
    pixel_size = 1;
    color_type = PNG_COLOR_TYPE_GRAY;
  } else {
    pixel_size = 3;
    color_type = PNG_COLOR_TYPE_RGB;
  }

  fp = fopen (path, "wb");
  if (!fp) {
    goto fopen_failed;
  }

  png_ptr = png_create_write_struct (PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
  if (png_ptr == NULL) {
    goto png_create_write_struct_failed;
  }

  info_ptr = png_create_info_struct (png_ptr);
  if (info_ptr == NULL) {
    goto png_create_info_struct_failed;
  }

  /** Set up error handling. */

  if (setjmp (png_jmpbuf (png_ptr))) {
    goto png_failure;
  }

  /** Set image attributes. */

  png_set_IHDR (png_ptr,
      info_ptr,
      bitmap->width,
      bitmap->height,
      depth,
      color_type,
      PNG_INTERLACE_NONE,
      PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT);

  /** Initialize rows of PNG. */

  row_pointers = png_malloc (png_ptr, bitmap->height * sizeof (png_byte *));
  for (y = 0; y < bitmap->height; y++) {
    png_byte *row =
        png_malloc (png_ptr, sizeof (uint8_t) * bitmap->width * pixel_size);
    row_pointers[y] = row;
    for (x = 0; x < bitmap->width; x++) {
      pixel_t *pixel = pixel_at (bitmap, x, y);
      if (bitmap->color_format == GRAY8) {
        *row++ = pixel->gray;
      } else {
        *row++ = pixel->red;
        *row++ = pixel->green;
        *row++ = pixel->blue;
      }
    }
  }

  /** Write the image data to "fp". */

  png_init_io (png_ptr, fp);
  png_set_rows (png_ptr, info_ptr, row_pointers);
  png_write_png (png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);

  /**
   * The routine has successfully written the file, so we set
   * "status" to a value which indicates success.
   */

  status = 0;

  for (y = 0; y < bitmap->height; y++) {
    png_free (png_ptr, row_pointers[y]);
  }
  png_free (png_ptr, row_pointers);

png_failure:
png_create_info_struct_failed:
  png_destroy_write_struct (&png_ptr, &info_ptr);
png_create_write_struct_failed:
  fclose (fp);
fopen_failed:
  return status;
}

/**
 * @brief The main function, provide filename of a bmp file as the 1st argument.
 */
int
main (int argc, char *argv[])
{
  const char option_gray[] = "--GRAY8";
  FILE *bmpF;
  bitmap_t bmp;
  int x, y;
  uint16_t width, height, *ptr16;
  size_t size;
  char byte;
  char header[26];              /** gen24bBMP.py gives you 24B headered bmp file */
  int ret;
  char *pngfilename;
  int strn;

  /** Read the .bmp file (argv[1]) */
  if (argc < 2 || argc > 3) {
    printf ("Usage: %s BMPfilename [OPTION:--GRAY8]\n\n", argv[0]);
    return 1;
  }
  strn = strlen (argv[1]);
  if (strn < 5 || argv[1][strn - 4] != '.' || argv[1][strn - 3] != 'b' ||
      argv[1][strn - 2] != 'm' || argv[1][strn - 1] != 'p') {
    printf ("The BMPfilename must be ending with \".bmp\"\n\n");
  }
  /** Check the option, --GRAY8 */
  strn = strlen (option_gray);
  if ((argc == 3) && (strlen (argv[2]) == strn)
      && (!strncmp (option_gray, argv[2], strn))) {
    bmp.color_format = GRAY8;
  } else {
    bmp.color_format = RGB;
  }

  bmpF = fopen (argv[1], "rb");
  if (!bmpF) {
    printf ("Cannot open the file: %s\n", argv[1]);
    return 2;
  }

  size = fread (header, 1, 26, bmpF);
  if (size != 26) {
    printf ("Cannot read the header from the file: %s\n", argv[1]);
    fclose (bmpF);
    return 3;
  }

  ptr16 = (uint16_t *) & (header[18]);
  width = *ptr16;
  ptr16 = (uint16_t *) & (header[20]);
  height = *ptr16;

  /** Let's not accept BMP files larger than 10000 x 10000 (Fix Covertify Issue #1029514) */
  if (width > 10000 || height > 10000 || width < 4 || height < 4) {
    printf
        ("We do not accept BMP files with height or width larger than 10000.\n");
    fclose (bmpF);
    return 100;
  }
  bmp.width = width;
  bmp.height = height;

  bmp.pixels = calloc (width * height, sizeof (pixel_t));

  for (y = (int) height - 1; y >= 0; y--) {
    for (x = 0; x < (int) width; x++) {
      pixel_t *pixel = pixel_at (&bmp, x, y);
      if (bmp.color_format == GRAY8) {
        uint8_t gray;
        size = fread (&gray, 1, 1, bmpF);
        if (size != 1) {
          printf ("x = %d / y = %d / (%d,%d) / size = %zu\n", x, y, width,
              height, size);
          goto error;
        }
        pixel->gray = gray;
      } else {
        uint8_t bgr[3];
        size = fread (bgr, 1, 3, bmpF);
        if (size != 3) {
          printf ("x = %d / y = %d / (%d,%d) / size = %zu\n", x, y, width,
              height, size);
          goto error;
        }
        pixel->red = bgr[2];
        pixel->green = bgr[1];
        pixel->blue = bgr[0];
      }
    }
    for (x = 0; x < (width * 3) % 4; x++) {
      size = fread (&byte, 1, 1, bmpF);
      /** Go through zero padding */
      if (size != 1)
        goto error;
    }
  }
  fclose (bmpF);

  pngfilename = g_strdup (argv[1]);

  /** Assume the last 4 characters are ".bmp" */
  strncpy (pngfilename + strlen (argv[1]) - 4, ".png", 5);
  ret = save_png_to_file (&bmp, pngfilename);
  free (bmp.pixels);

  return ret;
error:
  printf ("File read size error.\n");
  free (bmp.pixels);
  fclose (bmpF);
  return 10;
}
