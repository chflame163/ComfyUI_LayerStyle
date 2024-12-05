/**
 * File: DZ_widgets.js
 * Project: comfy_DZ
 * Author: Mel Massadian
 *
 * Copyright (c) 2023 Mel Massadian
 *
 */

import { app } from '../../scripts/app.js'
import { api } from '../../scripts/api.js'

import parseCss from './dz_parse-css.js'
import * as shared from './dz_comfy_shared.js'
import { log } from './dz_comfy_shared.js'

const newTypes = [, /*'BOOL'*/ 'COLOR', 'BBOX']

const withFont = (ctx, font, cb) => {
  const oldFont = ctx.font
  ctx.font = font
  cb()
  ctx.font = oldFont
}

const calculateTextDimensions = (ctx, value, width, fontSize = 16) => {
  const words = value.split(' ')
  const lines = []
  let currentLine = ''
  for (const word of words) {
    const testLine = currentLine.length === 0 ? word : `${currentLine} ${word}`
    const testWidth = ctx.measureText(testLine).width
    if (testWidth > width) {
      lines.push(currentLine)
      currentLine = word
    } else {
      currentLine = testLine
    }
  }
  if (lines.length === 0) lines.push(value)
  const textHeight = (lines.length + 1) * fontSize
  const maxLineWidth = lines.reduce(
    (maxWidth, line) => Math.max(maxWidth, ctx.measureText(line).width),
    0
  )
  return { textHeight, maxLineWidth }
}

export const DZWidgets = {
  BBOX: (key, val) => {
    /** @type {import("./types/litegraph").IWidget} */
    const widget = {
      name: key,
      type: 'BBOX',
      // options: val,
      y: 0,
      value: val?.default || [0, 0, 0, 0],
      options: {},

      draw: function (ctx, node, widget_width, widgetY, height) {
        const hide = this.type !== 'BBOX' && app.canvas.ds.scale > 0.5

        const show_text = true
        const outline_color = LiteGraph.WIDGET_OUTLINE_COLOR
        const background_color = LiteGraph.WIDGET_BGCOLOR
        const text_color = LiteGraph.WIDGET_TEXT_COLOR
        const secondary_text_color = LiteGraph.WIDGET_SECONDARY_TEXT_COLOR
        const H = LiteGraph.NODE_WIDGET_HEIGHT

        let margin = 15
        let numWidgets = 4 // Number of stacked widgets

        if (hide) return

        for (let i = 0; i < numWidgets; i++) {
          let currentY = widgetY + i * (H + margin) // Adjust Y position for each widget

          ctx.textAlign = 'left'
          ctx.strokeStyle = outline_color
          ctx.fillStyle = background_color
          ctx.beginPath()
          if (show_text)
            ctx.roundRect(margin, currentY, widget_width - margin * 2, H, [
              H * 0.5,
            ])
          else ctx.rect(margin, currentY, widget_width - margin * 2, H)
          ctx.fill()
          if (show_text) {
            if (!this.disabled) ctx.stroke()
            ctx.fillStyle = text_color
            if (!this.disabled) {
              ctx.beginPath()
              ctx.moveTo(margin + 16, currentY + 5)
              ctx.lineTo(margin + 6, currentY + H * 0.5)
              ctx.lineTo(margin + 16, currentY + H - 5)
              ctx.fill()
              ctx.beginPath()
              ctx.moveTo(widget_width - margin - 16, currentY + 5)
              ctx.lineTo(widget_width - margin - 6, currentY + H * 0.5)
              ctx.lineTo(widget_width - margin - 16, currentY + H - 5)
              ctx.fill()
            }
            ctx.fillStyle = secondary_text_color
            ctx.fillText(
              this.label || this.name,
              margin * 2 + 5,
              currentY + H * 0.7
            )
            ctx.fillStyle = text_color
            ctx.textAlign = 'right'

            ctx.fillText(
              Number(this.value).toFixed(
                this.options?.precision !== undefined
                  ? this.options.precision
                  : 3
              ),
              widget_width - margin * 2 - 20,
              currentY + H * 0.7
            )
          }
        }
      },
      mouse: function (event, pos, node) {
        let old_value = this.value
        let x = pos[0] - node.pos[0]
        let y = pos[1] - node.pos[1]
        let width = node.size[0]
        let H = LiteGraph.NODE_WIDGET_HEIGHT
        let margin = 5
        let numWidgets = 4 // Number of stacked widgets

        for (let i = 0; i < numWidgets; i++) {
          let currentY = y + i * (H + margin) // Adjust Y position for each widget

          if (
            event.type == LiteGraph.pointerevents_method + 'move' &&
            this.type == 'BBOX'
          ) {
            if (event.deltaX)
              this.value += event.deltaX * 0.1 * (this.options?.step || 1)
            if (this.options.min != null && this.value < this.options.min) {
              this.value = this.options.min
            }
            if (this.options.max != null && this.value > this.options.max) {
              this.value = this.options.max
            }
          } else if (event.type == LiteGraph.pointerevents_method + 'down') {
            let values = this.options?.values
            if (values && values.constructor === Function) {
              values = this.options.values(w, node)
            }
            let values_list = null

            let delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0
            if (this.type == 'BBOX') {
              this.value += delta * 0.1 * (this.options.step || 1)
              if (this.options.min != null && this.value < this.options.min) {
                this.value = this.options.min
              }
              if (this.options.max != null && this.value > this.options.max) {
                this.value = this.options.max
              }
            } else if (delta) {
              //clicked in arrow, used for combos
              let index = -1
              this.last_mouseclick = 0 //avoids dobl click event
              if (values.constructor === Object)
                index = values_list.indexOf(String(this.value)) + delta
              else index = values_list.indexOf(this.value) + delta
              if (index >= values_list.length) {
                index = values_list.length - 1
              }
              if (index < 0) {
                index = 0
              }
              if (values.constructor === Array) this.value = values[index]
              else this.value = index
            }
          } //end mousedown
          else if (
            event.type == LiteGraph.pointerevents_method + 'up' &&
            this.type == 'BBOX'
          ) {
            let delta = x < 40 ? -1 : x > widget_width - 40 ? 1 : 0
            if (event.click_time < 200 && delta == 0) {
              this.prompt(
                'Value',
                this.value,
                function (v) {
                  // check if v is a valid equation or a number
                  if (/^[0-9+\-*/()\s]+|\d+\.\d+$/.test(v)) {
                    try {
                      //solve the equation if possible
                      v = eval(v)
                    } catch (e) {}
                  }
                  this.value = Number(v)
                  shared.inner_value_change(this, this.value, event)
                }.bind(w),
                event
              )
            }
          }

          if (old_value != this.value)
            setTimeout(
              function () {
                shared.inner_value_change(this, this.value, event)
              }.bind(this),
              20
            )

          app.canvas.setDirty(true)
        }
      },
      computeSize: function (width) {
        return [width, LiteGraph.NODE_WIDGET_HEIGHT * 4]
      },
      // onDrawBackground: function (ctx) {
      //     if (!this.flags.collapsed) return;
      //     this.inputEl.style.display = "block";
      //     this.inputEl.style.top = this.graphcanvas.offsetTop + this.pos[1] + "px";
      //     this.inputEl.style.left = this.graphcanvas.offsetLeft + this.pos[0] + "px";
      // },
      // onInputChange: function (e) {
      //     const property = e.target.dataset.property;
      //     const bbox = this.getInputData(0);
      //     if (!bbox) return;
      //     bbox[property] = parseFloat(e.target.value);
      //     this.setOutputData(0, bbox);
      // }
    }

    widget.desc = 'Represents a Bounding Box with x, y, width, and height.'
    return widget
  },

  COLOR: (key, val, compute = false) => {
    /** @type {import("/types/litegraph").IWidget} */
    const widget = {}
    widget.y = 0
    widget.name = key
    widget.type = 'COLOR'
    widget.options = { default: '#ff0000' }
    widget.value = val || '#ff0000'
    widget.draw = function (ctx, node, widgetWidth, widgetY, height) {
      const hide = this.type !== 'COLOR' && app.canvas.ds.scale > 0.5
      if (hide) {
        return
      }
      const border = 3
      ctx.fillStyle = '#000'
      ctx.fillRect(0, widgetY, widgetWidth, height)
      ctx.fillStyle = this.value
      ctx.fillRect(
        border,
        widgetY + border,
        widgetWidth - border * 2,
        height - border * 2
      )
      const color = parseCss(this.value.default || this.value)
      if (!color) {
        return
      }
      ctx.fillStyle = shared.isColorBright(color.values, 125) ? '#000' : '#fff'

      ctx.font = '14px Arial'
      ctx.textAlign = 'center'
      ctx.fillText(this.name, widgetWidth * 0.5, widgetY + 14)
    }
    widget.mouse = function (e, pos, node) {
      if (e.type === 'pointerdown') {
        const widgets = node.widgets.filter((w) => w.type === 'COLOR')

        for (const w of widgets) {
          // color picker
          const rect = [w.last_y, w.last_y + 32]
          if (pos[1] > rect[0] && pos[1] < rect[1]) {
            const picker = document.createElement('input')
            picker.type = 'color'
            picker.value = this.value

            picker.style.position = 'absolute'
            picker.style.left = '999999px' //(window.innerWidth / 2) + "px";
            picker.style.top = '999999px' //(window.innerHeight / 2) + "px";

            document.body.appendChild(picker)

            picker.addEventListener('change', () => {
              this.value = picker.value
              node.graph._version++
              node.setDirtyCanvas(true, true)
              picker.remove()
            })

            picker.click()
          }
        }
      }
    }
    widget.computeSize = function (width) {
      return [width, 32]
    }

    return widget
  },

//  DEBUG_IMG: (name, val) => {
//    const w = {
//      name,
//      type: 'image',
//      value: val,
//      draw: function (ctx, node, widgetWidth, widgetY, height) {
//        const [cw, ch] = this.computeSize(widgetWidth)
//        shared.offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, ch)
//      },
//      computeSize: function (width) {
//        const ratio = this.inputRatio || 1
//        if (width) {
//          return [width, width / ratio + 4]
//        }
//        return [128, 128]
//      },
//      onRemoved: function () {
//        if (this.inputEl) {
//          this.inputEl.remove()
//        }
//      },
//    }
//
//    w.inputEl = document.createElement('img')
//    w.inputEl.src = w.value
//    w.inputEl.onload = function () {
//      w.inputRatio = w.inputEl.naturalWidth / w.inputEl.naturalHeight
//    }
//    document.body.appendChild(w.inputEl)
//    return w
//  },
//  DEBUG_STRING: (name, val) => {
//    const fontSize = 16
//    const w = {
//      name,
//      type: 'debug_text',
//
//      draw: function (ctx, node, widgetWidth, widgetY, height) {
//        // const [cw, ch] = this.computeSize(widgetWidth)
//        shared.offsetDOMWidget(this, ctx, node, widgetWidth, widgetY, height)
//      },
//      computeSize(width) {
//        if (!this.value) {
//          return [32, 32]
//        }
//        if (!width) {
//          console.debug(`No width ${this.parent.size}`)
//        }
//        let dimensions
//        withFont(app.ctx, `${fontSize}px monospace`, () => {
//          dimensions = calculateTextDimensions(app.ctx, this.value, width)
//        })
//        const widgetWidth = Math.max(
//          width || this.width || 32,
//          dimensions.maxLineWidth
//        )
//        const widgetHeight = dimensions.textHeight * 1.5
//        return [widgetWidth, widgetHeight]
//      },
//      onRemoved: function () {
//        if (this.inputEl) {
//          this.inputEl.remove()
//        }
//      },
//      get value() {
//        return this.inputEl.innerHTML
//      },
//      set value(val) {
//        this.inputEl.innerHTML = val
//        this.parent?.setSize?.(this.parent?.computeSize())
//      },
//    }
//
//    w.inputEl = document.createElement('p')
//    w.inputEl.style = `
//      text-align: center;
//      font-size: ${fontSize}px;
//      color: var(--input-text);
//      line-height: 0;
//      font-family: monospace;
//    `
//    w.value = val
//    document.body.appendChild(w.inputEl)
//
//    return w
//  },
}

/**
 * @returns {import("./types/comfy").ComfyExtension} extension
 */
const DZ_widgets = {
  name: 'DZ.widgets',

  init: async () => {
    log('Registering DZ.widgets')
//    try {
//      const res = await api.fetchApi('/DZ/debug')
//      const msg = await res.json()
//      if (!window.DZ) {
//        window.DZ = {}
//      }
//      window.DZ.DEBUG = msg.enabled
//    } catch (e) {
//      console.error('Error:', error)
//    }
  },

  setup: () => {
//    app.ui.settings.addSetting({
//      id: 'DZ.Debug.enabled',
//      name: '[DZ] Enable Debug (py and js)',
//      type: 'boolean',
//      defaultValue: false,
//
//      tooltip:
//        'This will enable debug messages in the console and in the python console respectively',
//      attrs: {
//        style: {
//          fontFamily: 'monospace',
//        },
//      },
//      async onChange(value) {
//        if (value) {
//          console.log('Enabled DEBUG mode')
//        }
//        if (!window.DZ) {
//          window.DZ = {}
//        }
//        window.DZ.DEBUG = value
//        await api
//          .fetchApi('/DZ/debug', {
//            method: 'POST',
//            body: JSON.stringify({
//              enabled: value,
//            }),
//          })
//          .then((response) => {})
//          .catch((error) => {
//            console.error('Error:', error)
//          })
//      },
//    })
  },

  getCustomWidgets: function () {
    return {
      BOOL: (node, inputName, inputData, app) => {
        console.debug('Registering bool')

        return {
          widget: node.addCustomWidget(
            DZWidgets.BOOL(inputName, inputData[1]?.default || false)
          ),
          minWidth: 150,
          minHeight: 30,
        }
      },

      COLOR: (node, inputName, inputData, app) => {
        console.debug('Registering color')
        return {
          widget: node.addCustomWidget(
            DZWidgets.COLOR(inputName, inputData[1]?.default || '#ff0000')
          ),
          minWidth: 150,
          minHeight: 30,
        }
      },
      // BBOX: (node, inputName, inputData, app) => {
      //     console.debug("Registering bbox")
      //     return {
      //         widget: node.addCustomWidget(DZWidgets.BBOX(inputName, inputData[1]?.default || [0, 0, 0, 0])),
      //         minWidth: 150,
      //         minHeight: 30,
      //     }

      // }
    }
  },
  /**
   * @param {import("./types/comfy").NodeType} nodeType
   * @param {import("./types/comfy").NodeDef} nodeData
   * @param {import("./types/comfy").App} app
   */
  async beforeRegisterNodeDef(nodeType, nodeData, app) {
    // const rinputs = nodeData.input?.required

    let has_custom = false
    if (nodeData.input && nodeData.input.required) {
      for (const i of Object.keys(nodeData.input.required)) {
        const input_type = nodeData.input.required[i][0]

        if (newTypes.includes(input_type)) {
          has_custom = true
          break
        }
      }
    }
    if (has_custom) {
      //- Add widgets on node creation
      const onNodeCreated = nodeType.prototype.onNodeCreated
      nodeType.prototype.onNodeCreated = function () {
        const r = onNodeCreated
          ? onNodeCreated.apply(this, arguments)
          : undefined
        this.serialize_widgets = true
        this.setSize?.(this.computeSize())

        this.onRemoved = function () {
          // When removing this node we need to remove the input from the DOM
          shared.cleanupNode(this)
        }
        return r
      }

      //- Extra menus
      const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions
      nodeType.prototype.getExtraMenuOptions = function (_, options) {
        const r = origGetExtraMenuOptions
          ? origGetExtraMenuOptions.apply(this, arguments)
          : undefined
        if (this.widgets) {
          let toInput = []
          let toWidget = []
          for (const w of this.widgets) {
            if (w.type === shared.CONVERTED_TYPE) {
              //- This is already handled by widgetinputs.js
              // toWidget.push({
              //     content: `Convert ${w.name} to widget`,
              //     callback: () => shared.convertToWidget(this, w),
              // });
            } else if (newTypes.includes(w.type)) {
              const config = nodeData?.input?.required[w.name] ||
                nodeData?.input?.optional?.[w.name] || [w.type, w.options || {}]

              toInput.push({
                content: `Convert ${w.name} to input`,
                callback: () => shared.convertToInput(this, w, config),
              })
            }
          }
          if (toInput.length) {
            options.push(...toInput, null)
          }

          if (toWidget.length) {
            options.push(...toWidget, null)
          }
        }

        return r
      }
    }

    //- Extending Python Nodes
    switch (nodeData.name) {
      case 'Psd Save (DZ)': {
        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (
          type,
          index,
          connected,
          link_info
        ) {
          const r = onConnectionsChange
            ? onConnectionsChange.apply(this, arguments)
            : undefined
          shared.dynamic_connection(this, index, connected)
          return r
        }
        break
      }
      //TODO: remove this non sense
      case 'Get Batch From History (DZ)': {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
          const r = onNodeCreated
            ? onNodeCreated.apply(this, arguments)
            : undefined
          const internal_count = this.widgets.find(
            (w) => w.name === 'internal_count'
          )
          shared.hideWidgetForGood(this, internal_count)
          internal_count.afterQueued = function () {
            this.value++
          }

          return r
        }

        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
          const r = onExecuted ? onExecuted.apply(this, message) : undefined
          return r
        }

        break
      }
      case 'Save Gif (DZ)':
      case 'Save Animated Image (DZ)': {
        const onExecuted = nodeType.prototype.onExecuted
        nodeType.prototype.onExecuted = function (message) {
          const prefix = 'anything_'
          const r = onExecuted ? onExecuted.apply(this, message) : undefined

          if (this.widgets) {
            const pos = this.widgets.findIndex((w) => w.name === `${prefix}_0`)
            if (pos !== -1) {
              for (let i = pos; i < this.widgets.length; i++) {
                this.widgets[i].onRemoved?.()
              }
              this.widgets.length = pos
            }

            let imgURLs = []
            if (message) {
              if (message.gif) {
                imgURLs = imgURLs.concat(
                  message.gif.map((params) => {
                    return api.apiURL(
                      '/view?' + new URLSearchParams(params).toString()
                    )
                  })
                )
              }
              if (message.apng) {
                imgURLs = imgURLs.concat(
                  message.apng.map((params) => {
                    return api.apiURL(
                      '/view?' + new URLSearchParams(params).toString()
                    )
                  })
                )
              }
              let i = 0
//              for (const img of imgURLs) {
//                const w = this.addCustomWidget(
//                  DZWidgets.DEBUG_IMG(`${prefix}_${i}`, img)
//                )
//                w.parent = this
//                i++
//              }
            }
            const onRemoved = this.onRemoved
            this.onRemoved = () => {
              shared.cleanupNode(this)
              return onRemoved?.()
            }
          }
          this.setSize?.(this.computeSize())
          return r
        }

        break
      }
      case 'Animation Builder (DZ)': {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
          const r = onNodeCreated
            ? onNodeCreated.apply(this, arguments)
            : undefined

          this.changeMode(LiteGraph.ALWAYS)

          const raw_iteration = this.widgets.find(
            (w) => w.name === 'raw_iteration'
          )
          const raw_loop = this.widgets.find((w) => w.name === 'raw_loop')

          const total_frames = this.widgets.find(
            (w) => w.name === 'total_frames'
          )
          const loop_count = this.widgets.find((w) => w.name === 'loop_count')

          shared.hideWidgetForGood(this, raw_iteration)
          shared.hideWidgetForGood(this, raw_loop)

          raw_iteration._value = 0

//          const value_preview = this.addCustomWidget(
//            DZWidgets['DEBUG_STRING']('value_preview', 'Idle')
//          )
//          value_preview.parent = this

//          const loop_preview = this.addCustomWidget(
//            DZWidgets['DEBUG_STRING']('loop_preview', 'Iteration: Idle')
//          )
//          loop_preview.parent = this

          const onReset = () => {
            raw_iteration.value = 0
            raw_loop.value = 0

            value_preview.value = 'Idle'
            loop_preview.value = 'Iteration: Idle'

            app.canvas.setDirty(true)
          }

          const reset_button = this.addWidget(
            'button',
            `Reset`,
            'reset',
            onReset
          )

          const run_button = this.addWidget('button', `Queue`, 'queue', () => {
            onReset() // this could maybe be a setting or checkbox
            app.queuePrompt(0, total_frames.value * loop_count.value)
            window.DZ?.notify?.(
              `Started a queue of ${total_frames.value} frames (for ${
                loop_count.value
              } loop, so ${total_frames.value * loop_count.value})`,
              5000
            )
          })

          this.onRemoved = () => {
            shared.cleanupNode(this)
            app.canvas.setDirty(true)
          }

          raw_iteration.afterQueued = function () {
            this.value++
            raw_loop.value = Math.floor(this.value / total_frames.value)

            value_preview.value = `frame: ${
              raw_iteration.value % total_frames.value
            } / ${total_frames.value - 1}`

            if (raw_loop.value + 1 > loop_count.value) {
              loop_preview.value = 'Done 😎!'
            } else {
              loop_preview.value = `current loop: ${raw_loop.value + 1}/${
                loop_count.value
              }`
            }
          }

          return r
        }

        break
      }
      case 'Text Encore Frames (DZ)': {
        const onConnectionsChange = nodeType.prototype.onConnectionsChange
        nodeType.prototype.onConnectionsChange = function (
          type,
          index,
          connected,
          link_info
        ) {
          const r = onConnectionsChange
            ? onConnectionsChange.apply(this, arguments)
            : undefined

          shared.dynamic_connection(this, index, connected)
          return r
        }
        break
      }
      case 'Interpolate Clip Sequential (DZ)': {
        const onNodeCreated = nodeType.prototype.onNodeCreated
        nodeType.prototype.onNodeCreated = function () {
          const r = onNodeCreated
            ? onNodeCreated.apply(this, arguments)
            : undefined
          const addReplacement = () => {
            const input = this.addInput(
              `replacement_${this.widgets.length}`,
              'STRING',
              ''
            )
            console.log(input)
            this.addWidget('STRING', `replacement_${this.widgets.length}`, '')
          }
          //- add
          this.addWidget('button', '+', 'add', function (value, widget, node) {
            console.log('Button clicked', value, widget, node)
            addReplacement()
          })
          //- remove
          this.addWidget(
            'button',
            '-',
            'remove',
            function (value, widget, node) {
              console.log(`Button clicked: ${value}`, widget, node)
            }
          )

          return r
        }
        break
      }
      case 'Styles Loader (DZ)': {
        const origGetExtraMenuOptions = nodeType.prototype.getExtraMenuOptions
        nodeType.prototype.getExtraMenuOptions = function (_, options) {
          const r = origGetExtraMenuOptions
            ? origGetExtraMenuOptions.apply(this, arguments)
            : undefined

          const getStyle = async (node) => {
            try {
              const getStyles = await api.fetchApi('/DZ/actions', {
                method: 'POST',
                body: JSON.stringify({
                  name: 'getStyles',
                  args:
                    node.widgets && node.widgets[0].value
                      ? node.widgets[0].value
                      : '',
                }),
              })

              const output = await getStyles.json()
              return output?.result
            } catch (e) {
              console.error(e)
            }
          }
          const extracters = [
            {
              content: 'Extract Positive to Text node',
              callback: async () => {
                const style = await getStyle(this)
                if (style && style.length >= 1) {
                  if (style[0]) {
                    window.DZ?.notify?.(
                      `Extracted positive from ${this.widgets[0].value}`
                    )
                    const tn = LiteGraph.createNode('Text box')
                    app.graph.add(tn)
                    tn.title = `${this.widgets[0].value} (Positive)`
                    tn.widgets[0].value = style[0]
                  } else {
                    window.DZ?.notify?.(
                      `No positive to extract for ${this.widgets[0].value}`
                    )
                  }
                }
              },
            },
            {
              content: 'Extract Negative to Text node',
              callback: async () => {
                const style = await getStyle(this)
                if (style && style.length >= 2) {
                  if (style[1]) {
                    window.DZ?.notify?.(
                      `Extracted negative from ${this.widgets[0].value}`
                    )
                    const tn = LiteGraph.createNode('Text box')
                    app.graph.add(tn)
                    tn.title = `${this.widgets[0].value} (Negative)`
                    tn.widgets[0].value = style[1]
                  } else {
                    window.DZ.notify(
                      `No negative to extract for ${this.widgets[0].value}`
                    )
                  }
                }
              },
            },
          ]
          options.push(...extracters)
        }

        break
      }
      case 'Save Tensors (DZ)': {
        const onDrawBackground = nodeType.prototype.onDrawBackground
        nodeType.prototype.onDrawBackground = function (ctx, canvas) {
          const r = onDrawBackground
            ? onDrawBackground.apply(this, arguments)
            : undefined
          // // draw a circle on the top right of the node, with text inside
          // ctx.fillStyle = "#fff";
          // ctx.beginPath();
          // ctx.arc(this.size[0] - this.node_width * 0.5, this.size[1] - this.node_height * 0.5, this.node_width * 0.5, 0, Math.PI * 2);
          // ctx.fill();

          // ctx.fillStyle = "#000";
          // ctx.textAlign = "center";
          // ctx.font = "bold 12px Arial";
          // ctx.fillText("Save Tensors", this.size[0] - this.node_width * 0.5, this.size[1] - this.node_height * 0.5);

          return r
        }
        break
      }
      default: {
        break
      }
    }
  },
}

app.registerExtension(DZ_widgets)
