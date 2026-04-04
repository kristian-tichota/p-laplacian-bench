;; -*- lexical-binding: t; -*-

(TeX-add-style-hook
 "packages"
 (lambda ()
   (TeX-add-to-alist 'LaTeX-provided-package-options
                     '(("inputenc" "utf8") ("fontenc" "T1") ("babel" "english") ("amsmath" "") ("amssymb" "") ("graphicx" "") ("hyperref" "") ("microtype" "")))
   (TeX-run-style-hooks
    "inputenc"
    "fontenc"
    "babel"
    "amsmath"
    "amssymb"
    "graphicx"
    "hyperref"
    "microtype"))
 :latex)

