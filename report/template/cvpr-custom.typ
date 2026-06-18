/**
 * cvpr-custom.typ
 *
 * Consolidated and corrected template for the VCS Report.
 * Replaces both cvpr2022.typ and cvpr2025.typ.
 */

#let std-bibliography = bibliography 

#let conf-name = [CVPR]
#let conf-year = [2025]
#let notice = [CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.]
#let indent = h(12pt)
#let eg = emph[e.g.]
#let etal = emph[et~al]

#let font-family = ("Times New Roman", "CMU Serif", "Latin Modern Roman", "New Computer Modern", "Libertinus Serif")
#let font-family-sans = ("Arial", "TeX Gyre Heros", "New Computer Modern Sans", "CMU Sans Serif", "DejaVu Sans")
#let font-family-mono = ("CMU Typewriter Text", "Latin Modern Mono", "New Computer Modern Mono", "DejaVu Sans Mono")
#let font-family-link = ("Courier New", "Nimbus Mono PS") + font-family-mono

#let font-size = (
  normal: 10pt,
  small: 9pt,
  footnote: 8pt,
  script: 7pt,
  tiny: 5pt,
  large: 12pt,
  Large: 14pt,
  LARGE: 17pt,
  huge: 20pt,
  Huge: 25pt,
)

#let color = (
  ref: rgb(100%, 0%, 0%),
  link: rgb(100%, 0%, 100%),
)

// Helper for Heading 3 run-in style
#let heading3(title, body) = [
  #v(11pt, weak: true)
  #text(size: 10pt, weight: "bold")[#title.] #body
]

#let format-affilation(affl) = {
  let lines = ()
  if "department" in affl { lines.push(affl.department) }
  if "institution" in affl { lines.push(affl.institution) }
  let address = ()
  if "location" in affl { address.push(affl.location) }
  if "country" in affl { address.push(affl.country) }
  if address.len() > 0 { lines.push(address.join([, ])) }
  lines.join([\ ])
}

#let format-author(author, affls) = box(baseline: 100%, {
  author.name
  if "affl" in author {
    [\ ]
    author.affl.map(it => format-affilation(affls.at(it))).join([\ ])
  }
  if "email" in author {
    show raw: set text(font: font-family-link, size: font-size.small, fill: black)
    v(9pt, weak: true)
    link(author.email, raw(author.email))
  }
})

#let dtcite(key) = context {
  if target() == "html" {
    html.elem(
      "dt-cite",
      attrs: (key: key),
    )
  } else {
    [@#key]
  }
}

// Distill-like track wrappers for tables/figures.
// HTML: maps to Distill classes.
// PDF: approximates expansion with Typst spacing primitives.
#let distill-track(track, content) = context {
  if target() == "html" {
    let cls = if track == "body" {
      "l-body"
    } else if track == "outset" {
      "l-body-outset"
    } else if track == "page" {
      "l-page"
    } else if track == "screen" {
      "l-screen-inset"
    } else {
      "l-body"
    }
    html.elem("div", attrs: (class: cls))[#content]
  } else {
    if track == "body" {
      block(width: 100%, content)
    } else if track == "outset" {
      pad(x: -1.5cm, block(width: 100% + 3cm, content))
    } else if track == "page" {
      pad(x: -2.8cm, block(width: 100% + 5.6cm, content))
    } else if track == "screen" {
      place(dx: -3.5cm, block(width: 21cm, content))
    } else {
      block(width: 100%, content)
    }
  }
}

#let distill-body(content) = distill-track("body", content)
#let distill-outset(content) = distill-track("outset", content)
#let distill-page(content) = distill-track("page", content)
#let distill-screen(content) = distill-track("screen", content)

#let make-title(title, authors, affls, id, accepted) = {
  block(width: 100%, spacing: 0pt, {
    set align(center)
    set text(size: font-size.Large, weight: "bold")
    v(0.375in) // Ensures title baseline is exactly 1-3/8 inches from top edge
    title
  })
  v(30pt, weak: true)

  block(width: 100%, spacing: 0pt, {
    set align(center + top)
    set text(size: font-size.large)
    if accepted != none and not accepted {
      [Anonymous CVPR submission\ ]
      [\ ]
      [Paper ID #id]
    } else {
      pad(left: 10pt, right: 12pt, {
        authors.map(it => format-author(it, affls)).join(h(0.5in))
      })
    }
  })
  v(34.5pt, weak: true)
}

#let render-distill(
  title: [],
  authors: (),
  affls: (),
  abstract: [],
  body,
) = {
  let distill-fallback-css = "body{margin:0;padding:0;font-family:Georgia,Times,serif;line-height:1.6;color:#111;background:#fff;}dt-article,section[role='doc-bibliography'],section[role='doc-endnotes']{display:block;box-sizing:border-box;width:min(100%,860px);margin-left:auto;margin-right:auto;padding-left:1.25rem;padding-right:1.25rem;}dt-article{padding-top:2.25rem;padding-bottom:3rem;}section[role='doc-bibliography'],section[role='doc-endnotes']{padding-bottom:2rem;}dt-article h1{text-align:center;font-size:2.1rem;line-height:1.2;margin:0 0 1.25rem;}dt-byline{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:1rem;margin:0 0 1.4rem;}dt-byline h3{font-size:0.95rem;text-transform:uppercase;letter-spacing:.03em;margin:0 0 .25rem;color:#333;}dt-byline p{margin:.2rem 0 0;font-size:.95rem;color:#444;}dt-article > div:first-of-type p{margin:0 0 1.5rem;font-size:1rem;}dt-article figure{margin:1.2rem 0 1.4rem;}dt-article figure > *{max-width:100%;}dt-article svg,dt-article canvas,dt-article img{max-width:100%;height:auto;}dt-article table{width:100%;border-collapse:collapse;font-size:0.88em;line-height:1.35;}dt-article th{font-weight:600;border-bottom:1.4px solid rgba(0,0,0,0.6);padding:6px 8px;text-align:left;}dt-article td{border-bottom:1px solid rgba(0,0,0,0.12);padding:6px 8px;vertical-align:top;}dt-article tr:last-child td{border-bottom:1.4px solid rgba(0,0,0,0.6);}dt-article figure[id] table{margin-top:0.2rem;}"
  let distill-front-matter = (
    "authors:\n"
    + authors.map(author => "- " + author.name).join("\n")
    + "\naffiliations:\n"
    + authors.map(author => {
      if "affl" in author and author.affl.len() > 0 {
        "- \"" + author.affl.map(id => format-affilation(affls.at(id))).join(", ") + "\""
      } else {
        "- \"\""
      }
    }).join("\n")
  )

  [
    #html.elem("style")[#distill-fallback-css]

    #html.elem(
      "script",
      attrs: (
        src: "https://distill.pub/template.v1.js",
      ),
    )

    #html.elem("script", attrs: (type: "text/front-matter"))[#distill-front-matter]

    #html.elem("dt-article", attrs: (class: "centered"))[
      #html.elem("h1")[#title]

      #html.elem("dt-byline")[]

      #html.elem("div")[
        #html.elem("p")[
          #strong[Abstract.] 
          #abstract
        ]
      ]

      #{
        show math.equation: set block(
          above: 1.2em,
          below: 1.2em,
        )

        body
      }
    ]
  ]
}

#let maybe-render-distill(renderer, title, authors, affls, abstract, body) = context {
  if renderer == "distill" and target() == "html" {
    render-distill(
      title: title,
      authors: authors,
      affls: affls,
      abstract: abstract,
      body,
    )
  } else {
    none
  }
}

#let cvpr_report(
  title: [],
  authors: (),
  keywords: (),
  date: auto,
  abstract: [],
  bibliography: none,
  appendix: none,
  accepted: false,
  id: none,
  starting-page: 1, // Parameter for your assigned starting page
  paper-size: "us-letter", // Supports dynamic switching between "us-letter" and "a4"
  renderer: "cvpr",
  body,
) = {
  let (authors, affls) = if authors.len() == 2 { authors } else { ((), ()) }
  let distill-render = maybe-render-distill(renderer, title, authors, affls, abstract, body)
  if distill-render != none { return distill-render }
  if accepted != none and not accepted {
    authors = ((name: "Anonymous Author"), )
  }
  if id == none { id = "******" }

  let page-width = if paper-size == "a4" { 21.0cm } else { 8.5in }
  let page-height = if paper-size == "a4" { 29.7cm } else { 11in }

  set document(
    title: title,
    author: authors.map(it => it.name).join(", ", last: " and "),
    keywords: keywords,
    date: date
  )

  set page(
    paper: paper-size,
    // Ensures a perfect print area of 17.5cm x 22.54cm with a 1.0 inch top margin
    margin: (
      left: (page-width - 17.5cm) / 2,
      right: (page-width - 17.5cm) / 2,
      top: 1in,
      bottom: page-height - 1in - 22.54cm,
    ),
    header: if accepted != none and not accepted {
      set align(center)
      set text(font: font-family-sans, size: font-size.footnote, fill: rgb(50%, 50%, 100%))
      strong[#conf-name #conf-year Submission \##id. CONFIDENTIAL REVIEW COPY. DO NOT DISTRIBUTE.]
    },
    // Dynamically ensures the page number text lands precisely 0.75 inches from the bottom edge
    footer-descent: (page-height - 1in - 22.54cm) - 0.75in,
    footer: context {
      let ix = counter(page).get().first()
      align(center, text(size: font-size.normal, font: font-family, [#ix]))
    },
  )

  // Explicitly apply starting page number sequence immediately
  counter(page).update(starting-page)

  set text(font: font-family, size: font-size.normal)
  set par(first-line-indent: 0.166666in, leading: 0.532em, spacing: 0.54em, justify: true)
  show raw: set text(font: font-family-mono, size: font-size.normal)

  // Configure headings
  set heading(numbering: "1.1.")
  show heading.where(level: 1): it => {
    set text(size: font-size.large, weight: "bold") // 12pt
    set block(above: 12pt, below: 12pt)
    it
  }
  show heading.where(level: 2): it => {
    set text(size: 11pt, weight: "bold") // 11pt
    set block(above: 11pt, below: 11pt)
    it
  }
  // Heading 3 is blocked structure-wise to prevent accidental line-breaks
  show heading.where(level: 3): it => block(text(size: 10pt, weight: "bold")[#it.body.])

  set math.equation(numbering: "(1)", supplement: [Eq.])
  show math.equation: set block(spacing: 9pt)

  show footnote.entry: set text(size: font-size.footnote) // 8pt
  set footnote.entry(
    separator: line(length: 1.3in, stroke: 0.35pt),
    clearance: 6.65pt,
    gap: 0.40em,
    indent: 12pt
  )

  // Figures & Captions
  set figure(gap: 12pt)
  set figure.caption(separator: [.])
  show figure.caption: set text(size: font-size.small) // 9pt
  show figure.caption: set align(center) // Allows natural short caption centring

  // References and Links
  show link: set text(font: font-family-link, fill: color.link)
  show ref: it => {
    let el = it.element
    if el == none { return it }
    let supplement = if it.supplement != auto { it.supplement } else { el.supplement }

    if el.func() == math.equation {
      show link: set text(font: font-family, fill: color.ref)
      let cnt = counter(math.equation)
      let ix = numbering("1", ..cnt.at(el.location()))
      let href = link(el.location(), ix)
      [#supplement~(#href)]
    } else if el.func() == heading {
      show link: set text(font: font-family, fill: color.ref)
      let cnt = counter(heading)
      
      let num-format = if el.numbering != none { el.numbering } else { "1.1." }
      let ix = numbering(num-format, ..cnt.at(el.location()))
      
      let href = link(el.location(), ix)
      [#supplement~#href]
    } else if el.func() == figure {
      show link: set text(font: font-family, fill: color.ref)
      let cnt = counter(figure.where(kind: el.kind))
      
      // FIX: Check if the target figure resides inside an appendix section 
      // by inspecting the nearest preceding heading's formatting pattern.
      let ix = context {
        let before-headings = query(selector(heading).before(el.location()))
        let is-appendix = if before-headings.len() > 0 {
          before-headings.last().numbering == "A.1"
        } else {
          false
        }
        
        if is-appendix {
          let heading-num = counter(heading).at(el.location()).at(0)
          let letter = numbering("A", heading-num)
          let fig-num = cnt.at(el.location()).at(0)
          [#letter.#fig-num]
        } else {
          numbering(el.numbering, ..cnt.at(el.location()))
        }
      }
      
      let href = link(el.location(), ix)
      [#supplement~#href]
    } else {
      it
    }
  }

  make-title(title, authors, affls, id, accepted)

  // Two-column layout configuration
  columns(2, gutter: 0.3125in, { // Exactly 5/16 inch separation
    // Render Abstract
    block(width: 100%, {
      set par(first-line-indent: 0pt)
      align(center, text(size: font-size.large)[*Abstract*]) // 12pt Bold Centred
      v(12pt, weak: true)
      emph[#abstract] // 10pt Italicised Justified
      v(24pt, weak: true) // Leaves two blank lines after abstract
    })

    body 

    if bibliography != none {
      set std-bibliography(title: [References], style: "ieee.csl")
      show std-bibliography: set text(size: font-size.small) // 9pt Single-Spaced References
      bibliography
    }
  })

  if appendix != none {
    set heading(numbering: "A.1", supplement: [Appendix])
    counter(heading).update(0)
    
    // FIX: Reset figure counters so appendix visualisations start fresh at 1
    counter(figure.where(kind: image)).update(0)
    counter(figure.where(kind: table)).update(0)

    // FIX: Contextually inject the active appendix letter prefix into figure captions
    set figure(numbering: (..args) => context {
      let sec = counter(heading).get().at(0)
      let sec_letter = numbering("A", sec)
      [#sec_letter.#args.pos().at(0)]
    })
    
    show heading.where(level: 1): it => block(width: 100%, below: 12pt, above: 12pt)[
      #text(size: font-size.large, weight: "bold")[Appendix #counter(heading).display(it.numbering) #h(0.5em) #it.body]
    ]
    
    appendix
  }
}