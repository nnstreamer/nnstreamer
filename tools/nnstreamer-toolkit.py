#!/usr/bin/env python

##
# NNStreamer Toolkit
# Copyright (C) 2018 Samsung Electronics
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Library General Public
# License as published by the Free Software Foundation;
# version 2.1 of the License.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Library General Public License for more details.

##
# @file   nnstreamer-toolkit.py
# @brief  A front-end GUI toolkit for tracing, debugging, and profiling NNStreamer.
# @author Geunsik Lim <geunsik.lim@samsung.com>
# @date   03 Dec 2018
# @bug    No known bugs
# @note   TODO/NYI: Interconnect a front-end GUI applicaiton and back-end tools.

import gtk
import pango

##
# @brief A GTK-based front-end GUI application
class PyApp(gtk.Window):
   def __init__(self):
      super(PyApp, self).__init__()
      self.set_title("NNStreamer Toolkits")
      self.set_default_size(500, 400)
      self.set_position(gtk.WIN_POS_CENTER)
      
      mb = gtk.MenuBar()

      # Tracing menu
      menu1 = gtk.Menu()
      tracing = gtk.MenuItem("_Tracing")
      tracing.set_submenu(menu1)

      msg_gstshark = gtk.MenuItem("GstShark")
      menu1.append(msg_gstshark)

      # Debugging menu
      menu2 = gtk.Menu()
      debugging = gtk.MenuItem("_Debugging")
      debugging.set_submenu(menu2)

      msg_gstdebug = gtk.MenuItem("GST__DEBUG")
      menu2.append(msg_gstdebug)

      msg_gstdebugdump = gtk.MenuItem("GST__DEBUG__DUMP__DOT__DIR")
      menu2.append(msg_gstdebugdump)

      msg_gstdebugger = gtk.MenuItem("gst-debugger")
      menu2.append(msg_gstdebugger)

      # Profiling menu
      menu3 = gtk.Menu()
      profiling = gtk.MenuItem("_Profiling")
      profiling.set_submenu(menu3)

      msg_gstinstruments = gtk.MenuItem("gst-instruments")
      menu3.append(msg_gstinstruments)

      msg_hawktracer = gtk.MenuItem("HawkTracer")
      menu3.append(msg_hawktracer)

      mb.append(tracing)
      mb.append(debugging)
      mb.append(profiling)

      # Help menu
      menu4 = gtk.Menu()
      hlp = gtk.MenuItem("_Help")
      hlp.set_submenu(menu4)

      color = gtk.MenuItem("Color widget")
      menu4.append(color)

      abt = gtk.MenuItem("About")
      menu4.append(abt)

      mb.append(hlp)

 
      vbox = gtk.VBox(False, 2)
      vbox.pack_start(mb, False, False, 0)
      self.add(vbox)
      self.text = gtk.Label("")
      self.text.set_markup("<b>      NNStreamer toolkit provides tracing, debugging, and profiling tool \
                             \n      in order that developers can deploy NNStreamer on their own devices.</b>")
      vbox.pack_start(self.text, True, True, 0)

      msg_gstshark.connect("activate",self.on_gstshark)
      msg_gstdebug.connect("activate",self.on_gstdebug)
      msg_gstdebugdump.connect("activate",self.on_gstdebugdump)
      msg_gstdebugger.connect("activate",self.on_gstdebugger)
      msg_gstinstruments.connect("activate",self.on_gstinstruments)
      msg_hawktracer.connect("activate",self.on_hawktracer)
      color.connect("activate",self.on_color)
      abt.connect("activate",self.on_abtdlg)
      
      self.connect("destroy", gtk.main_quit)
      self.show_all()

   def on_gstshark(self, widget):
      # MessageDialog usage code
      md = gtk.MessageDialog(self,
         gtk.DIALOG_DESTROY_WITH_PARENT, gtk.MESSAGE_ERROR,
         gtk.BUTTONS_CLOSE, "Welcome to GstShark.")
      md.run()
      md.destroy()

   def on_gstdebug(self, widget):
      # MessageDialog usage code
      md = gtk.MessageDialog(self,
         gtk.DIALOG_DESTROY_WITH_PARENT, gtk.MESSAGE_ERROR,
         gtk.BUTTONS_CLOSE, "Welcome to GST_DEBUG.")
      md.run()
      md.destroy()

   def on_gstdebugdump(self, widget):
      # MessageDialog usage code
      md = gtk.MessageDialog(self,
         gtk.DIALOG_DESTROY_WITH_PARENT, gtk.MESSAGE_ERROR,
         gtk.BUTTONS_CLOSE, "Welcome to GST_DEBUG_DUMP_DOT_DIR.")
      md.run()
      md.destroy()

   def on_gstdebugger(self, widget):
      # MessageDialog usage code
      md = gtk.MessageDialog(self,
         gtk.DIALOG_DESTROY_WITH_PARENT, gtk.MESSAGE_ERROR,
         gtk.BUTTONS_CLOSE, "Welcome to gst-debugger.")
      md.run()
      md.destroy()

   def on_gstinstruments(self, widget):
      # MessageDialog usage code
      md = gtk.MessageDialog(self,
         gtk.DIALOG_DESTROY_WITH_PARENT, gtk.MESSAGE_ERROR,
         gtk.BUTTONS_CLOSE, "Welcome to gst-instruments.")
      md.run()
      md.destroy()

   def on_hawktracer(self, widget):
      # MessageDialog usage code
      md = gtk.MessageDialog(self,
         gtk.DIALOG_DESTROY_WITH_PARENT, gtk.MESSAGE_ERROR,
         gtk.BUTTONS_CLOSE, "Welcome to HawkTracer.")
      md.run()
      md.destroy()

   def on_color(self, widget):
      #Color Chooser Dialog usage cde
      dlg = gtk.ColorSelectionDialog("Select color")
      col = dlg.run()
      sel = dlg.colorsel.get_current_color()
      self.text.modify_fg(gtk.STATE_NORMAL, sel)

   def on_abtdlg(self, widget):
      #About Dialog usage code
      about = gtk.AboutDialog()
      about.set_program_name("NNStreamer Toolkit")
      about.set_version("0.0.1")
      about.set_authors([
            'Geunsik Lim',
            'Bug Reports and Patches:',
               '   MyungJoo Ham',
               '   Jijoong Moon',
               '   Sangjung Woo',
               '   Wook Song',
               '   Jaeyun Jung',
               '   Hyoungjoo Ahn',
            ])

      about.set_copyright("(c) Samsung Electronics")
      about.set_comments("About NNStreamer Toolkit")
      about.set_website("https://github.com/nnstreamer/nnstreamer")
      about.run()
      about.destroy()

if __name__ == '__main__':
   PyApp()
   gtk.main()
