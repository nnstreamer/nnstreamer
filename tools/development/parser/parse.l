%{
/* SPDX-License-Identifier: LGPL-2.1-only */
/**
 * This is imported from GStreamer and altered to parse GST-Pipeline
 */

#include <math.h>
#include <string.h>

#include <glib/gprintf.h>
#include <glib/glib.h>

#include "types.h"
#include "grammar.tab.h"

#ifdef malloc
#undef malloc
#endif

#ifdef free
#undef free
#endif

#ifdef realloc
#undef realloc
#endif

#define malloc g_malloc
#define free g_free
#define realloc g_realloc

/* Override the default ECHO so as to avoid fortify warnings. Ignore the
   embedded-NUL case for now. We know yytext is NUL-terminated. */
#define ECHO g_fprintf(yyout, "%s", yytext)

#define PRINT(...) g_printinfo (__VA_ARGS__)

%}

_operator [(){}.!:,;=]
_identifier [[:alnum:]_][[:alnum:]\-_%:]*

_char ("\\".)|([^[:space:]])
_string {_char}+|("\""([^\"]|"\\\"")*"\"")|("'"([^']|"\\\'")*"'")

_assign [[:space:]]*"="[[:space:]]*

_protocol [[:alpha:]][[:alnum:]+-\.]*
_url ({_protocol}"://"{_string}|["."{_identifier}]?"/"{_string})|({_protocol}"://")

/* we must do this here, because nearly everything matches a {_string} */
_assignment {_identifier}{_assign}{_string}

/* get pad/element references and stuff with dots right */
_padref "."{_identifier}
_ref {_identifier}"."{_identifier}?
_binref {_identifier}[[:space:]]*"."[[:space:]]*"("

/* links */
_mimechar [[:alnum:]-]
_mimetype {_mimechar}+"/"{_mimechar}+
_capschar ("\\".)|([^\;!])
_capsstring {_capschar}+
_caps {_mimetype}(","[^!]|{_capsstring})*
_link ([!:][[:space:]]*{_caps}([[:space:]]*(";"[[:space:]]*{_caps})*[[:space:]]*)*[!:])|([!:])

%x value
%option noyywrap
%option nounput
%option reentrant
%option bison-bridge
%option never-interactive
%option noinput
%%

{_assignment} {
    /* "=" */
    PRINT ("ASSIGNMENT: %s", yytext);
    yylval->ss = g_strdup (yytext);
    BEGIN (INITIAL);
    return ASSIGNMENT;
}

{_padref} {
    yytext++;
    PRINT ("PADREF: %s", yytext);
    yylval->ss = g_strdup (yytext);
    BEGIN (INITIAL);
    return PADREF;
}

{_ref} {
    PRINT ("REF: %s", yytext);
    yylval->ss = g_strdup (yytext);
    BEGIN (INITIAL);
    return REF;
}

{_binref} {
    gchar *pos = yytext;
    while (!g_ascii_isspace (*pos) && (*pos != '.')) pos++;
    *pos = '\0';
    PRINT ("BINREF: %s", yytext);
    yylval->ss = g_strdup (yytext);
    BEGIN (INITIAL);
    return BINREF;
}

{_identifier} {
    PRINT ("IDENTIFIER: %s", yytext);
    yylval->ss = g_strdup (yytext);
    BEGIN (INITIAL);
    return IDENTIFIER;
}

{_link} {
    gchar *c = yytext;
    gchar op;
    gboolean link_all;

    PRINT ("LINK: %s", yytext);
    /* First char is a link operator */
    link_all = (*c == ':');
    c++;
    if (*c) {
      while (g_ascii_isspace (*c)) c++;
      c = yylval->ss = g_strdup (c);
      while (*c) c++;
      /* Last non-space char is a link operator */
      op = *--c;
      if (op != '!' && op != ':')
	g_assert_not_reached ();
      if (op == ':')
        link_all = TRUE;

      /* Walk backward past whitespaces - what remains
       * is a filter caps string, or empty */
      while (g_ascii_isspace (*--c));
      *++c = '\0';
    } else {
      yylval->ss = NULL;
    }
    BEGIN (INITIAL);
    return link_all ? LINK_ALL : LINK;
}
{_url} {
  PRINT ("URL: %s", yytext);
  yylval->ss = g_strdup (yytext);
  gst_parse_unescape (yylval->ss);
  BEGIN (INITIAL);
  return PARSE_URL;
}

{_operator} { PRINT ("OPERATOR: [%s]", yytext); return *yytext; }

[[:space:]]+ { PRINT ("SPACE: [%s]", yytext); }

. {
    PRINT ("Invalid Lexer element: %s\n", yytext);
    return *yytext;
}

%%
