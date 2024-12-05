/**
 * File: dz_comfy_shared.js 
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

import { app } from '../../scripts/app.js'

export const log = (...args) => {
//  if (window.DZ?.DEBUG) {
//    console.debug(...args)
//  }
}

//- WIDGET UTILS
export const CONVERTED_TYPE = 'converted-widget'

export const hasWidgets = (node) => {
  if (!node.widgets || !node.widgets?.[Symbol.iterator]) {
    return false
  }
  return true
}

export const cleanupNode = (node) => {
  if (!hasWidgets(node)) {
    return
  }

  for (const w of node.widgets) {
    if (w.canvas) {
      w.canvas.remove()
    }
    if (w.inputEl) {
      w.inputEl.remove()
    }
    // calls the widget remove callback
    w.onRemoved?.()
  }
}

export function offsetDOMWidget(
  widget,
  ctx,
  node,
  widgetWidth,
  widgetY,
  height
) {
  const margin = 10
  const elRect = ctx.canvas.getBoundingClientRect()
  const transform = new DOMMatrix()
    .scaleSelf(
      elRect.width / ctx.canvas.width,
      elRect.height / ctx.canvas.height
    )
    .multiplySelf(ctx.getTransform())
    .translateSelf(margin, margin + widgetY)

  const scale = new DOMMatrix().scaleSelf(transform.a, transform.d)
  Object.assign(widget.inputEl.style, {
    transformOrigin: '0 0',
    transform: scale,
    left: `${transform.a + transform.e}px`,
    top: `${transform.d + transform.f}px`,
    width: `${widgetWidth - margin * 2}px`,
    // height: `${(widget.parent?.inputHeight || 32) - (margin * 2)}px`,
    height: `${(height || widget.parent?.inputHeight || 32) - margin * 2}px`,

    position: 'absolute',
    background: !node.color ? '' : node.color,
    color: !node.color ? '' : 'white',
    zIndex: 5, //app.graph._nodes.indexOf(node),
  })
}

/**
 * Extracts the type and link type from a widget config object.
 * @param {*} config
 * @returns
 */
export function getWidgetType(config) {
  // Special handling for COMBO so we restrict links based on the entries
  let type = config?.[0]
  let linkType = type
  if (type instanceof Array) {
    type = 'COMBO'
    linkType = linkType.join(',')
  }
  return { type, linkType }
}

export const dynamic_connection = (
  node,
  index,
  connected,
  connectionPrefix = 'input_',
  connectionType = 'PSDLAYER'
) => {
  // remove all non connected inputs
  if (!connected && node.inputs.length > 1) {
    log(`Removing input ${index} (${node.inputs[index].name})`)
    if (node.widgets) {
      const w = node.widgets.find((w) => w.name === node.inputs[index].name)
      if (w) {
        w.onRemoved?.()
        node.widgets.length = node.widgets.length - 1
      }
    }
    node.removeInput(index)

    // make inputs sequential again
    for (let i = 0; i < node.inputs.length; i++) {
      node.inputs[i].label = `${connectionPrefix}${i + 1}`
    }
  }

  // add an extra input
  if (node.inputs[node.inputs.length - 1].link != undefined) {
    log(
      `Adding input ${node.inputs.length + 1} (${connectionPrefix}${
        node.inputs.length + 1
      })`
    )

    node.addInput(
      `${connectionPrefix}${node.inputs.length + 1}`,
      connectionType
    )
  }
}

/**
 * Appends a callback to the extra menu options of a given node type.
 * @param {*} nodeType
 * @param {*} cb
 */
export function addMenuHandler(nodeType, cb) {
  const getOpts = nodeType.prototype.getExtraMenuOptions
  nodeType.prototype.getExtraMenuOptions = function () {
    const r = getOpts.apply(this, arguments)
    cb.apply(this, arguments)
    return r
  }
}

export function hideWidget(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.hidden = true
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = CONVERTED_TYPE + suffix
  widget.serializeValue = () => {
    // Prevent serializing the widget if we have no input linked
    const { link } = node.inputs.find((i) => i.widget?.name === widget.name)
    if (link == null) {
      return undefined
    }
    return widget.origSerializeValue
      ? widget.origSerializeValue()
      : widget.value
  }

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidget(node, w, ':' + widget.name)
    }
  }
}

export function showWidget(widget) {
  widget.type = widget.origType
  widget.computeSize = widget.origComputeSize
  widget.serializeValue = widget.origSerializeValue

  delete widget.origType
  delete widget.origComputeSize
  delete widget.origSerializeValue

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      showWidget(w)
    }
  }
}

export function convertToWidget(node, widget) {
  showWidget(widget)
  const sz = node.size
  node.removeInput(node.inputs.findIndex((i) => i.widget?.name === widget.name))

  for (const widget of node.widgets) {
    widget.last_y -= LiteGraph.NODE_SLOT_HEIGHT
  }

  // Restore original size but grow if needed
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function convertToInput(node, widget, config) {
  hideWidget(node, widget)

  const { linkType } = getWidgetType(config)

  // Add input and store widget config for creating on primitive node
  const sz = node.size
  node.addInput(widget.name, linkType, {
    widget: { name: widget.name, config },
  })

  for (const widget of node.widgets) {
    widget.last_y += LiteGraph.NODE_SLOT_HEIGHT
  }

  // Restore original size but grow if needed
  node.setSize([Math.max(sz[0], node.size[0]), Math.max(sz[1], node.size[1])])
}

export function hideWidgetForGood(node, widget, suffix = '') {
  widget.origType = widget.type
  widget.origComputeSize = widget.computeSize
  widget.origSerializeValue = widget.serializeValue
  widget.computeSize = () => [0, -4] // -4 is due to the gap litegraph adds between widgets automatically
  widget.type = CONVERTED_TYPE + suffix
  // widget.serializeValue = () => {
  //     // Prevent serializing the widget if we have no input linked
  //     const w = node.inputs?.find((i) => i.widget?.name === widget.name);
  //     if (w?.link == null) {
  //         return undefined;
  //     }
  //     return widget.origSerializeValue ? widget.origSerializeValue() : widget.value;
  // };

  // Hide any linked widgets, e.g. seed+seedControl
  if (widget.linkedWidgets) {
    for (const w of widget.linkedWidgets) {
      hideWidgetForGood(node, w, ':' + widget.name)
    }
  }
}

export function fixWidgets(node) {
  if (node.inputs) {
    for (const input of node.inputs) {
      log(input)
      if (input.widget || node.widgets) {
        // if (newTypes.includes(input.type)) {
        const matching_widget = node.widgets.find((w) => w.name === input.name)
        if (matching_widget) {
          // if (matching_widget.hidden) {
          //     log(`Already hidden skipping ${matching_widget.name}`)
          //     continue
          // }
          const w = node.widgets.find((w) => w.name === matching_widget.name)
          if (w && w.type != CONVERTED_TYPE) {
            log(w)
            log(`hidding ${w.name}(${w.type}) from ${node.type}`)
            log(node)
            hideWidget(node, w)
          } else {
            log(`converting to widget ${w}`)

            convertToWidget(node, input)
          }
        }
      }
    }
  }
}
export function inner_value_change(widget, value, event = undefined) {
  if (widget.type == 'number' || widget.type == 'BBOX') {
    value = Number(value)
  } else if (widget.type == 'BOOL') {
    value = Boolean(value)
  }
  widget.value = value
  if (
    widget.options &&
    widget.options.property &&
    node.properties[widget.options.property] !== undefined
  ) {
    node.setProperty(widget.options.property, value)
  }
  if (widget.callback) {
    widget.callback(widget.value, app.canvas, node, pos, event)
  }
}

//- COLOR UTILS
export function isColorBright(rgb, threshold = 240) {
  const brightess = getBrightness(rgb)
  return brightess > threshold
}

function getBrightness(rgbObj) {
  return Math.round(
    (parseInt(rgbObj[0]) * 299 +
      parseInt(rgbObj[1]) * 587 +
      parseInt(rgbObj[2]) * 114) /
      1000
  )
}

//- HTML / CSS UTILS
export function defineClass(className, classStyles) {
  const styleSheets = document.styleSheets

  // Helper function to check if the class exists in a style sheet
  function classExistsInStyleSheet(styleSheet) {
    const rules = styleSheet.rules || styleSheet.cssRules
    for (const rule of rules) {
      if (rule.selectorText === `.${className}`) {
        return true
      }
    }
    return false
  }

  // Check if the class is already defined in any of the style sheets
  let classExists = false
  for (const styleSheet of styleSheets) {
    if (classExistsInStyleSheet(styleSheet)) {
      classExists = true
      break
    }
  }

  // If the class doesn't exist, add the new class definition to the first style sheet
  if (!classExists) {
    if (styleSheets[0].insertRule) {
      styleSheets[0].insertRule(`.${className} { ${classStyles} }`, 0)
    } else if (styleSheets[0].addRule) {
      styleSheets[0].addRule(`.${className}`, classStyles, 0)
    }
  }
}
