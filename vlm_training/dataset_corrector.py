"""PySide6 Dataset Annotation Corrector for Vision-Language Models.

Allows loading JSONL datasets with bounding box annotations, inspecting the
images and boxes, and modifying, creating, or deleting bounding boxes using
the mouse and keyboard.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from PySide6.QtCore import (
    QObject,
    QPoint,
    QPointF,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import (
    QAction,
    QBrush,
    QColor,
    QFont,
    QKeySequence,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
    QWheelEvent,
)
from shiboken6 import isValid
from PySide6.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QCheckBox,
    QFileDialog,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSceneHoverEvent,
    QGraphicsSceneMouseEvent,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTextEdit,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

# ---------------------------------------------------------------------------
# Data Models and Helpers
# ---------------------------------------------------------------------------

PALETTE_COLORS = [
    QColor("#3B82F6"),  # Blue
    QColor("#10B981"),  # Emerald
    QColor("#F59E0B"),  # Amber
    QColor("#EF4444"),  # Red
    QColor("#8B5CF6"),  # Purple
    QColor("#EC4899"),  # Pink
    QColor("#06B6D4"),  # Cyan
    QColor("#84CC16"),  # Lime
    QColor("#F97316"),  # Orange
    QColor("#14B8A6"),  # Teal
]


def color_for_label(label: str) -> QColor:
    """Deterministically assign a pleasant color for a given label string."""
    hash_val = sum(ord(c) for c in label)
    return PALETTE_COLORS[hash_val % len(PALETTE_COLORS)]


@dataclass
class BBox:
    x1: int
    y1: int
    x2: int
    y2: int
    label: str = "product"

    def normalized(self) -> BBox:
        nx1, nx2 = min(self.x1, self.x2), max(self.x1, self.x2)
        ny1, ny2 = min(self.y1, self.y2), max(self.y1, self.y2)
        return BBox(
            x1=max(0, min(1000, round(nx1))),
            y1=max(0, min(1000, round(ny1))),
            x2=max(0, min(1000, round(nx2))),
            y2=max(0, min(1000, round(ny2))),
            label=self.label,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox_2d": [self.x1, self.y1, self.x2, self.y2],
            "label": self.label,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BBox:
        bbox_coords = data.get("bbox_2d", [0, 0, 0, 0])
        label = data.get("label", "product")
        if len(bbox_coords) == 4:
            x1, y1, x2, y2 = bbox_coords
            return cls(int(x1), int(y1), int(x2), int(y2), str(label)).normalized()
        return cls(0, 0, 0, 0, str(label))


def parse_gpt_turn(content: str) -> list[BBox]:
    """Parse list of bounding boxes from GPT response."""
    # Look for ```json ... ``` code blocks first
    code_block_match = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", content, re.DOTALL)
    if code_block_match:
        json_str = code_block_match.group(1)
    else:
        # Look for raw array
        bracket_match = re.search(r"\[\s*\{.*?\}\s*\]", content, re.DOTALL)
        json_str = bracket_match.group(0) if bracket_match else content.strip()

    try:
        items = json.loads(json_str)
        if isinstance(items, list):
            bboxes = []
            for item in items:
                if isinstance(item, dict):
                    bboxes.append(BBox.from_dict(item))
            return bboxes
    except Exception:
        pass
    return []


def serialize_bboxes_to_gpt(bboxes: list[BBox]) -> str:
    """Format bboxes list into ```json ... ``` format."""
    dicts = [b.to_dict() for b in bboxes]
    json_text = json.dumps(dicts)
    return f"```json\n{json_text}\n```"


# ---------------------------------------------------------------------------
# Custom Interactive Bounding Box Graphics Item
# ---------------------------------------------------------------------------

HANDLE_SIZE = 8
HANDLE_HIT_MARGIN = 6

HANDLE_NONE = 0
HANDLE_TOP_LEFT = 1
HANDLE_TOP = 2
HANDLE_TOP_RIGHT = 3
HANDLE_RIGHT = 4
HANDLE_BOTTOM_RIGHT = 5
HANDLE_BOTTOM = 6
HANDLE_BOTTOM_LEFT = 7
HANDLE_LEFT = 8


class BBoxGraphicsItem(QGraphicsItem):
    """Interactive rectangle with resize handles, dragging, and label pill."""

    def __init__(
        self,
        bbox: BBox,
        img_width: int,
        img_height: int,
        parent: QGraphicsItem | None = None,
    ):
        super().__init__(parent)
        self.img_width = max(1, img_width)
        self.img_height = max(1, img_height)
        self.label = bbox.label
        self.color = color_for_label(self.label)

        # Scene pixel coordinates
        x1 = (bbox.x1 / 1000.0) * self.img_width
        y1 = (bbox.y1 / 1000.0) * self.img_height
        x2 = (bbox.x2 / 1000.0) * self.img_width
        y2 = (bbox.y2 / 1000.0) * self.img_height
        self._rect = QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))

        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemIsFocusable
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(1)

        self._active_handle = HANDLE_NONE
        self._drag_start_pos = QPointF()
        self._drag_start_rect = QRectF()
        self._is_dragging = False
        self._is_resizing = False

    def setSelected(self, selected: bool) -> None:
        if selected:
            sc = self.scene()
            if sc:
                for it in sc.selectedItems():
                    if it != self:
                        it.setSelected(False)
        super().setSelected(selected)
        self.setZValue(10 if selected else 1)
        self.update()

    def to_bbox(self) -> BBox:
        r = self._rect.normalized()
        x1 = round((r.left() / self.img_width) * 1000.0)
        y1 = round((r.top() / self.img_height) * 1000.0)
        x2 = round((r.right() / self.img_width) * 1000.0)
        y2 = round((r.bottom() / self.img_height) * 1000.0)
        return BBox(x1, y1, x2, y2, self.label).normalized()

    def set_rect(self, rect: QRectF) -> None:
        self.prepareGeometryChange()
        # Clamp within image bounds
        left = max(0.0, min(float(self.img_width), rect.left()))
        top = max(0.0, min(float(self.img_height), rect.top()))
        right = max(0.0, min(float(self.img_width), rect.right()))
        bottom = max(0.0, min(float(self.img_height), rect.bottom()))
        self._rect = QRectF(QPointF(left, top), QPointF(right, bottom)).normalized()
        self.update()

    def set_bbox_coords(self, x1: int, y1: int, x2: int, y2: int) -> None:
        px1 = (x1 / 1000.0) * self.img_width
        py1 = (y1 / 1000.0) * self.img_height
        px2 = (x2 / 1000.0) * self.img_width
        py2 = (y2 / 1000.0) * self.img_height
        self.set_rect(QRectF(QPointF(px1, py1), QPointF(px2, py2)))

    def set_label(self, label: str) -> None:
        self.label = label
        self.color = color_for_label(label)
        self.update()

    def boundingRect(self) -> QRectF:
        margin = HANDLE_SIZE + 10
        # Include space for label badge on top
        return self._rect.normalized().adjusted(-margin, -24, margin, margin)

    def shape(self) -> QPainterPath:
        path = QPainterPath()
        r = self._rect.normalized()
        path.addRect(r)
        badge_h = 22.0
        badge_y = r.top() - badge_h if r.top() >= badge_h else r.top()
        badge_rect = QRectF(r.left(), badge_y, min(120.0, max(50.0, r.width())), badge_h)
        path.addRect(badge_rect)
        return path

    def _get_handle_rects(self) -> dict[int, QRectF]:
        r = self._rect.normalized()
        s = HANDLE_SIZE
        hs = s / 2.0
        return {
            HANDLE_TOP_LEFT: QRectF(r.left() - hs, r.top() - hs, s, s),
            HANDLE_TOP: QRectF(r.center().x() - hs, r.top() - hs, s, s),
            HANDLE_TOP_RIGHT: QRectF(r.right() - hs, r.top() - hs, s, s),
            HANDLE_RIGHT: QRectF(r.right() - hs, r.center().y() - hs, s, s),
            HANDLE_BOTTOM_RIGHT: QRectF(r.right() - hs, r.bottom() - hs, s, s),
            HANDLE_BOTTOM: QRectF(r.center().x() - hs, r.bottom() - hs, s, s),
            HANDLE_BOTTOM_LEFT: QRectF(r.left() - hs, r.bottom() - hs, s, s),
            HANDLE_LEFT: QRectF(r.left() - hs, r.center().y() - hs, s, s),
        }

    def _handle_at(self, pos: QPointF) -> int:
        margin = HANDLE_HIT_MARGIN
        for handle, rect in self._get_handle_rects().items():
            if rect.adjusted(-margin, -margin, margin, margin).contains(pos):
                return handle
        return HANDLE_NONE

    def _cursor_for_handle(self, handle: int) -> Qt.CursorShape:
        mapping = {
            HANDLE_TOP_LEFT: Qt.CursorShape.SizeFDiagCursor,
            HANDLE_BOTTOM_RIGHT: Qt.CursorShape.SizeFDiagCursor,
            HANDLE_TOP_RIGHT: Qt.CursorShape.SizeBDiagCursor,
            HANDLE_BOTTOM_LEFT: Qt.CursorShape.SizeBDiagCursor,
            HANDLE_TOP: Qt.CursorShape.SizeVerCursor,
            HANDLE_BOTTOM: Qt.CursorShape.SizeVerCursor,
            HANDLE_LEFT: Qt.CursorShape.SizeHorCursor,
            HANDLE_RIGHT: Qt.CursorShape.SizeHorCursor,
        }
        return mapping.get(handle, Qt.CursorShape.ArrowCursor)

    def paint(self, painter: QPainter, option: Any, widget: Any = None) -> None:
        r = self._rect.normalized()
        is_selected = self.isSelected()

        painter.save()
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Fill interior
        fill_color = QColor(self.color)
        fill_color.setAlpha(65 if is_selected else 35)
        painter.setBrush(QBrush(fill_color))

        # 2. Outline stroke
        pen_color = QColor(self.color)
        if is_selected:
            pen = QPen(pen_color, 2.5, Qt.PenStyle.SolidLine)
        else:
            pen = QPen(pen_color, 1.8, Qt.PenStyle.SolidLine)
        painter.setPen(pen)
        painter.drawRect(r)

        # 3. Label badge
        badge_text = f"{self.label}"
        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        text_w = fm.horizontalAdvance(badge_text)
        text_h = fm.height()
        badge_w = text_w + 12
        badge_h = text_h + 4

        badge_y = r.top() - badge_h
        if badge_y < 0:
            badge_y = r.top() + 2
        badge_x = max(0.0, r.left())
        badge_rect = QRectF(badge_x, badge_y, badge_w, badge_h)

        # Draw badge background pill
        painter.setPen(Qt.PenStyle.NoPen)
        badge_bg = QColor(self.color.darker(120))
        badge_bg.setAlpha(225)
        painter.setBrush(QBrush(badge_bg))
        painter.drawRoundedRect(badge_rect, 3, 3)

        # Draw badge text
        painter.setPen(QPen(QColor("#FFFFFF")))
        painter.drawText(
            badge_rect,
            Qt.AlignmentFlag.AlignCenter,
            badge_text,
        )

        # 4. Handles (if selected)
        if is_selected:
            handle_pen = QPen(QColor("#FFFFFF"), 1.2)
            handle_brush = QBrush(self.color)
            painter.setPen(handle_pen)
            painter.setBrush(handle_brush)
            for h_rect in self._get_handle_rects().values():
                painter.drawRect(h_rect)

        painter.restore()

    def hoverMoveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        if self.isSelected():
            handle = self._handle_at(event.pos())
            if handle != HANDLE_NONE:
                self.setCursor(self._cursor_for_handle(handle))
                super().hoverMoveEvent(event)
                return
        if self._rect.normalized().contains(event.pos()):
            self.setCursor(Qt.CursorShape.SizeAllCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event: QGraphicsSceneHoverEvent) -> None:
        self.setCursor(Qt.CursorShape.ArrowCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            sc = self.scene()
            if sc:
                for it in sc.selectedItems():
                    if it != self:
                        it.setSelected(False)
            self.setSelected(True)

            handle = self._handle_at(event.pos())
            if handle != HANDLE_NONE:
                self._active_handle = handle
                self._is_resizing = True
                self._drag_start_pos = event.scenePos()
                self._drag_start_rect = QRectF(self._rect)
                event.accept()
                return
            if self.shape().contains(event.pos()) or self._rect.normalized().contains(event.pos()):
                self._is_dragging = True
                self._drag_start_pos = event.scenePos()
                self._drag_start_rect = QRectF(self._rect)
                event.accept()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._is_resizing:
            delta = event.scenePos() - self._drag_start_pos
            orig = self._drag_start_rect
            new_rect = QRectF(orig)

            if self._active_handle in (HANDLE_TOP_LEFT, HANDLE_LEFT, HANDLE_BOTTOM_LEFT):
                new_rect.setLeft(min(orig.right() - 5, orig.left() + delta.x()))
            if self._active_handle in (HANDLE_TOP_RIGHT, HANDLE_RIGHT, HANDLE_BOTTOM_RIGHT):
                new_rect.setRight(max(orig.left() + 5, orig.right() + delta.x()))
            if self._active_handle in (HANDLE_TOP_LEFT, HANDLE_TOP, HANDLE_TOP_RIGHT):
                new_rect.setTop(min(orig.bottom() - 5, orig.top() + delta.y()))
            if self._active_handle in (HANDLE_BOTTOM_LEFT, HANDLE_BOTTOM, HANDLE_BOTTOM_RIGHT):
                new_rect.setBottom(max(orig.top() + 5, orig.bottom() + delta.y()))

            self.set_rect(new_rect)
            sc: Any = self.scene()
            if sc is not None and hasattr(sc, "bbox_geometry_changed"):
                sc.bbox_geometry_changed.emit(self)
            event.accept()
            return

        if self._is_dragging:
            delta = event.scenePos() - self._drag_start_pos
            orig = self._drag_start_rect
            new_left = orig.left() + delta.x()
            new_top = orig.top() + delta.y()

            # Bound inside image
            new_left = max(0.0, min(float(self.img_width - orig.width()), new_left))
            new_top = max(0.0, min(float(self.img_height - orig.height()), new_top))

            self.set_rect(QRectF(new_left, new_top, orig.width(), orig.height()))
            sc = self.scene()
            if sc is not None and hasattr(sc, "bbox_geometry_changed"):
                cast(Any, sc).bbox_geometry_changed.emit(self)
            event.accept()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            was_interactive = self._is_resizing or self._is_dragging
            self._is_resizing = False
            self._is_dragging = False
            self._active_handle = HANDLE_NONE
            sc = self.scene()
            if was_interactive and sc is not None and hasattr(sc, "bbox_edit_committed"):
                cast(Any, sc).bbox_edit_committed.emit()
            event.accept()
            return
        super().mouseReleaseEvent(event)


# ---------------------------------------------------------------------------
# Annotation Graphics Scene & View
# ---------------------------------------------------------------------------

MODE_SELECT = 0
MODE_DRAW = 1


class AnnotationScene(QGraphicsScene):
    bbox_selected = Signal(BBoxGraphicsItem)
    bbox_deselected = Signal()
    bbox_geometry_changed = Signal(BBoxGraphicsItem)
    bbox_edit_committed = Signal()
    new_bbox_drawn = Signal(BBox)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self.pixmap_item: QGraphicsPixmapItem | None = None
        self.img_width = 0
        self.img_height = 0
        self.mode = MODE_SELECT
        self.default_label = "product"

        self._drawing = False
        self._draw_start = QPointF()
        self._temp_rect_item: QGraphicsRectItem | None = None

        self.selectionChanged.connect(self._on_selection_changed)

    def set_image(self, pixmap: QPixmap) -> None:
        self.clear()
        self.pixmap_item = self.addPixmap(pixmap)
        self.pixmap_item.setZValue(-100)
        self.img_width = pixmap.width()
        self.img_height = pixmap.height()
        self.setSceneRect(0, 0, self.img_width, self.img_height)

    def _on_selection_changed(self) -> None:
        selected = [it for it in self.selectedItems() if isinstance(it, BBoxGraphicsItem)]
        if selected:
            active = [it for it in selected if it.zValue() >= 10]
            pick = active[0] if active else selected[-1]
            self.bbox_selected.emit(pick)
        else:
            self.bbox_deselected.emit()

    def mousePressEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self.mode == MODE_DRAW and event.button() == Qt.MouseButton.LeftButton:
            # If user clicked on an existing box while in draw mode, pick that box
            clicked_boxes = [it for it in self.items(event.scenePos()) if isinstance(it, BBoxGraphicsItem)]
            if clicked_boxes:
                clicked_box = clicked_boxes[0]
                for it in self.selectedItems():
                    if it != clicked_box:
                        it.setSelected(False)
                clicked_box.setSelected(True)
                p = self.parent()
                if p is not None and hasattr(p, "set_mode"):
                    cast(Any, p).set_mode(MODE_SELECT)
                event.accept()
                return

            pos = event.scenePos()
            # Bound within scene image
            if 0 <= pos.x() <= self.img_width and 0 <= pos.y() <= self.img_height:
                self._drawing = True
                self._draw_start = pos
                pen = QPen(QColor("#3B82F6"), 2, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(59, 130, 246, 50))
                self._temp_rect_item = self.addRect(QRectF(pos, pos), pen, brush)
                self._temp_rect_item.setZValue(100)
                event.accept()
                return

        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drawing and self._temp_rect_item:
            pos = event.scenePos()
            clamped_x = max(0.0, min(float(self.img_width), pos.x()))
            clamped_y = max(0.0, min(float(self.img_height), pos.y()))
            cur_rect = QRectF(self._draw_start, QPointF(clamped_x, clamped_y)).normalized()
            self._temp_rect_item.setRect(cur_rect)
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QGraphicsSceneMouseEvent) -> None:
        if self._drawing and event.button() == Qt.MouseButton.LeftButton:
            self._drawing = False
            if self._temp_rect_item:
                rect = self._temp_rect_item.rect().normalized()
                self.removeItem(self._temp_rect_item)
                self._temp_rect_item = None

                # Only keep if reasonable size (e.g. >= 5px)
                if rect.width() >= 5 and rect.height() >= 5:
                    x1 = round((rect.left() / self.img_width) * 1000.0)
                    y1 = round((rect.top() / self.img_height) * 1000.0)
                    x2 = round((rect.right() / self.img_width) * 1000.0)
                    y2 = round((rect.bottom() / self.img_height) * 1000.0)
                    new_box = BBox(x1, y1, x2, y2, self.default_label).normalized()
                    self.new_bbox_drawn.emit(new_box)
            event.accept()
            return
        super().mouseReleaseEvent(event)


class AnnotationView(QGraphicsView):
    cursor_position_changed = Signal(int, int, int, int)  # px_x, px_y, norm_x, norm_y

    def __init__(self, scene: AnnotationScene, parent: QWidget | None = None):
        super().__init__(scene, parent)
        self._scene = scene
        self.setMinimumSize(800, 600)
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMouseTracking(True)

        self._is_panning = False
        self._pan_start_pos = QPoint()
        self._space_pressed = False

    def wheelEvent(self, event: QWheelEvent) -> None:
        factor = 1.15 if event.angleDelta().y() > 0 else 1.0 / 1.15
        self.scale(factor, factor)

    def keyPressEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = True
            self.setCursor(Qt.CursorShape.OpenHandCursor)
            event.accept()
            return
        if event.key() == Qt.Key.Key_C and not event.isAutoRepeat():
            win = self.window()
            if win is not None and hasattr(win, "start_add_box"):
                cast(Any, win).start_add_box()
                event.accept()
                return
        if (
            event.key() in (Qt.Key.Key_D, Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
            and not event.isAutoRepeat()
        ):
            win = self.window()
            if win is not None and hasattr(win, "delete_selected_bbox"):
                cast(Any, win).delete_selected_bbox()
                event.accept()
                return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event: Any) -> None:
        if event.key() == Qt.Key.Key_Space and not event.isAutoRepeat():
            self._space_pressed = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().keyReleaseEvent(event)

    def _event_pos(self, event: Any) -> QPoint:
        if hasattr(event, "position"):
            return cast(QPoint, event.position().toPoint())
        return cast(QPoint, event.pos())

    def mousePressEvent(self, event: Any) -> None:
        if (
            event.button() == Qt.MouseButton.MiddleButton
            or (event.button() == Qt.MouseButton.LeftButton and self._space_pressed)
        ):
            self._is_panning = True
            self._pan_start_pos = self._event_pos(event)
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: Any) -> None:
        event_pt = self._event_pos(event)
        # Emit cursor coordinates
        scene_pos = self.mapToScene(event_pt)
        px_x = round(scene_pos.x())
        px_y = round(scene_pos.y())
        if self._scene.img_width > 0 and self._scene.img_height > 0:
            norm_x = round((px_x / self._scene.img_width) * 1000.0)
            norm_y = round((px_y / self._scene.img_height) * 1000.0)
        else:
            norm_x, norm_y = 0, 0
        self.cursor_position_changed.emit(px_x, px_y, norm_x, norm_y)

        if self._is_panning:
            delta = event_pt - self._pan_start_pos
            self._pan_start_pos = event_pt
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - delta.x())
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - delta.y())
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: Any) -> None:
        if self._is_panning and (
            event.button() in (Qt.MouseButton.MiddleButton, Qt.MouseButton.LeftButton)
        ):
            self._is_panning = False
            if self._space_pressed:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def fit_image_in_view(self) -> None:
        if self._scene.img_width > 0 and self.viewport().width() > 20 and self.viewport().height() > 20:
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)


# ---------------------------------------------------------------------------
# Main Application Window
# ---------------------------------------------------------------------------

STYLESHEET = """
QMainWindow {
    background-color: #121215;
    color: #E2E8F0;
}
QWidget {
    font-family: 'Segoe UI', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
    font-size: 13px;
    color: #E2E8F0;
}
QToolBar {
    background: #1A1A22;
    border-bottom: 1px solid #2D2D3A;
    spacing: 6px;
    padding: 6px 10px;
}
QToolButton {
    background: #252532;
    border: 1px solid #363649;
    border-radius: 5px;
    padding: 5px 10px;
    color: #E2E8F0;
    font-weight: 500;
}
QToolButton:hover {
    background: #313144;
    border-color: #4F4F6A;
}
QToolButton:checked {
    background: #2563EB;
    border-color: #3B82F6;
    color: #FFFFFF;
}
QToolButton:disabled {
    background: #1A1A22;
    color: #555568;
    border-color: #2D2D3A;
}
QGroupBox {
    background-color: #1A1A22;
    border: 1px solid #2D2D3A;
    border-radius: 8px;
    margin-top: 14px;
    padding-top: 18px;
    padding-left: 10px;
    padding-right: 10px;
    padding-bottom: 10px;
    font-weight: 600;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    color: #94A3B8;
    padding: 0 4px;
}
QListWidget, QTableWidget {
    background-color: #16161D;
    border: 1px solid #2D2D3A;
    border-radius: 6px;
    color: #E2E8F0;
    selection-background-color: #2563EB;
    selection-color: #FFFFFF;
    outline: none;
}
QListWidget::item {
    padding: 6px 8px;
    border-radius: 4px;
    margin: 1px 2px;
}
QListWidget::item:hover {
    background-color: #232330;
}
QListWidget::item:selected {
    background-color: #2563EB;
}
QHeaderView::section {
    background-color: #1E1E28;
    border: 1px solid #2D2D3A;
    padding: 4px;
    font-weight: 600;
    color: #94A3B8;
}
QLineEdit, QSpinBox, QComboBox {
    background-color: #16161D;
    border: 1px solid #363649;
    border-radius: 5px;
    padding: 5px 8px;
    color: #FFFFFF;
}
QLineEdit:focus, QSpinBox:focus, QComboBox:focus {
    border: 1px solid #3B82F6;
}
QPushButton {
    background-color: #252532;
    border: 1px solid #363649;
    border-radius: 5px;
    padding: 6px 12px;
    color: #FFFFFF;
    font-weight: 500;
}
QPushButton:hover {
    background-color: #313144;
    border-color: #4F4F6A;
}
QPushButton:pressed {
    background-color: #1D1D28;
}
QPushButton#primaryBtn {
    background-color: #2563EB;
    border: 1px solid #3B82F6;
}
QPushButton#primaryBtn:hover {
    background-color: #1D4ED8;
}
QPushButton#dangerBtn {
    background-color: #991B1B;
    border: 1px solid #DC2626;
}
QPushButton#dangerBtn:hover {
    background-color: #B91C1C;
}
QSplitter::handle {
    background-color: #242432;
}
QTabWidget::pane {
    border: 1px solid #2D2D3A;
    background-color: #1A1A22;
    border-radius: 6px;
}
QTabBar::tab {
    background-color: #16161D;
    border: 1px solid #2D2D3A;
    padding: 6px 12px;
    color: #94A3B8;
    border-top-left-radius: 5px;
    border-top-right-radius: 5px;
    margin-right: 2px;
}
QTabBar::tab:selected {
    background-color: #1A1A22;
    color: #38BDF8;
    border-bottom: 1px solid #1A1A22;
}
QStatusBar {
    background-color: #121215;
    border-top: 1px solid #232330;
    color: #94A3B8;
}
QScrollBar:vertical {
    border: none;
    background: #14141A;
    width: 10px;
}
QScrollBar::handle:vertical {
    background: #323244;
    border-radius: 5px;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #4B4B66;
}
QScrollBar:horizontal {
    border: none;
    background: #14141A;
    height: 10px;
}
QScrollBar::handle:horizontal {
    background: #323244;
    border-radius: 5px;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #4B4B66;
}
"""


class DatasetCorrector(QMainWindow):
    def __init__(self, initial_jsonl_path: str | None = None):
        super().__init__()
        self.setWindowTitle("VLM Dataset Corrector")
        self.resize(1440, 920)
        self.setMinimumSize(1100, 700)
        self.setStyleSheet(STYLESHEET)
        self._first_show = True

        # Dataset State
        self.jsonl_path: Path | None = None
        self.items: list[dict[str, Any]] = []
        self.current_index: int = 0
        self.dirty_indices: set[int] = set()

        # Per-item edit history for Undo/Redo
        self.undo_stack: list[list[BBox]] = []
        self.redo_stack: list[list[BBox]] = []

        # Current Image & BBoxes
        self.current_pixmap: QPixmap | None = None
        self.current_bboxes: list[BBox] = []
        self.bbox_items: list[BBoxGraphicsItem] = []
        self.selected_bbox_item: BBoxGraphicsItem | None = None

        # Build UI
        self._init_scene_and_view()
        self._init_actions()
        self._init_toolbar()
        self._init_statusbar()
        self._init_sidebar()

        # Connect signals
        self._connect_signals()

        # Try loading initial file
        if initial_jsonl_path and os.path.exists(initial_jsonl_path):
            self.load_dataset(initial_jsonl_path)
        else:
            default_candidate = Path("example_dataset_item.jsonl")
            if default_candidate.exists():
                self.load_dataset(str(default_candidate.resolve()))

    # -----------------------------------------------------------------------
    # Setup UI Components
    # -----------------------------------------------------------------------

    def _init_scene_and_view(self) -> None:
        self.scene = AnnotationScene(self)
        self.view = AnnotationView(self.scene, self)

    def _init_actions(self) -> None:
        # File Actions
        self.act_open = QAction("Open JSONL...", self)
        self.act_open.setShortcut(QKeySequence.StandardKey.Open)
        self.act_open.triggered.connect(self.prompt_open_file)

        self.act_save = QAction("Save", self)
        self.act_save.setShortcut(QKeySequence.StandardKey.Save)
        self.act_save.triggered.connect(self.save_dataset)

        self.act_save_as = QAction("Save As...", self)
        self.act_save_as.setShortcut(QKeySequence("Ctrl+Shift+S"))
        self.act_save_as.triggered.connect(self.prompt_save_as)

        # Navigation
        self.act_prev = QAction("◀ Prev Item", self)
        self.act_prev.setShortcuts([QKeySequence("Left"), QKeySequence("PageUp")])
        self.act_prev.triggered.connect(self.prev_item)

        self.act_next = QAction("Next Item ▶", self)
        self.act_next.setShortcuts([QKeySequence("Right"), QKeySequence("PageDown")])
        self.act_next.triggered.connect(self.next_item)

        # Edit Actions
        self.act_undo = QAction("Undo", self)
        self.act_undo.setShortcut(QKeySequence.StandardKey.Undo)
        self.act_undo.triggered.connect(self.undo)

        self.act_redo = QAction("Redo", self)
        self.act_redo.setShortcut(QKeySequence.StandardKey.Redo)
        self.act_redo.triggered.connect(self.redo)

        self.act_add_box = QAction("Add Box", self)
        self.act_add_box.setShortcut(QKeySequence("C"))
        self.act_add_box.triggered.connect(self.start_add_box)

        self.act_delete = QAction("Delete Box", self)
        self.act_delete.setShortcuts([
            QKeySequence("D"),
            QKeySequence(QKeySequence.StandardKey.Delete),
            QKeySequence("Backspace"),
        ])
        self.act_delete.triggered.connect(self.delete_selected_bbox)

        self.act_duplicate = QAction("Duplicate Box", self)
        self.act_duplicate.setShortcut(QKeySequence("Ctrl+D"))
        self.act_duplicate.triggered.connect(self.duplicate_selected_bbox)

        # View Actions
        self.act_fit = QAction("Fit View", self)
        self.act_fit.setShortcut(QKeySequence("F"))
        self.act_fit.triggered.connect(self.view.fit_image_in_view)

        self.act_zoom_in = QAction("Zoom In", self)
        self.act_zoom_in.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.act_zoom_in.triggered.connect(lambda: self.view.scale(1.2, 1.2))

        self.act_zoom_out = QAction("Zoom Out", self)
        self.act_zoom_out.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.act_zoom_out.triggered.connect(lambda: self.view.scale(1.0 / 1.2, 1.0 / 1.2))

        # Add actions to window for global shortcuts
        for act in [
            self.act_open,
            self.act_save,
            self.act_save_as,
            self.act_prev,
            self.act_next,
            self.act_undo,
            self.act_redo,
            self.act_add_box,
            self.act_delete,
            self.act_duplicate,
            self.act_fit,
            self.act_zoom_in,
            self.act_zoom_out,
        ]:
            self.addAction(act)

    def _init_toolbar(self) -> None:
        toolbar = QToolBar("Main Toolbar", self)
        toolbar.setMovable(False)
        self.addToolBar(toolbar)

        toolbar.addAction(self.act_open)
        toolbar.addAction(self.act_save)
        toolbar.addSeparator()

        toolbar.addAction(self.act_prev)
        self.lbl_item_counter = QLabel(" 0 / 0 ")
        self.lbl_item_counter.setStyleSheet("font-weight: bold; padding: 0 6px;")
        toolbar.addWidget(self.lbl_item_counter)
        toolbar.addAction(self.act_next)
        toolbar.addSeparator()

        # Tools: Select vs Draw
        self.btn_select_mode = QPushButton("Select")
        self.btn_select_mode.setCheckable(True)
        self.btn_select_mode.setChecked(True)
        self.btn_select_mode.clicked.connect(lambda: self.set_mode(MODE_SELECT))
        toolbar.addWidget(self.btn_select_mode)

        self.btn_draw_mode = QPushButton("Draw Box")
        self.btn_draw_mode.setCheckable(True)
        self.btn_draw_mode.clicked.connect(lambda: self.set_mode(MODE_DRAW))
        toolbar.addWidget(self.btn_draw_mode)

        toolbar.addSeparator()
        toolbar.addWidget(QLabel("Default Label:"))
        self.edit_default_label = QLineEdit("product")
        self.edit_default_label.setFixedWidth(110)
        self.edit_default_label.textChanged.connect(self._on_default_label_changed)
        toolbar.addWidget(self.edit_default_label)

        toolbar.addSeparator()
        toolbar.addAction(self.act_undo)
        toolbar.addAction(self.act_redo)
        toolbar.addSeparator()

        toolbar.addAction(self.act_fit)
        toolbar.addAction(self.act_zoom_in)
        toolbar.addAction(self.act_zoom_out)

        # Auto-save checkbox
        self.chk_autosave = QCheckBox("Auto-save on switch")
        self.chk_autosave.setChecked(True)
        toolbar.addSeparator()
        toolbar.addWidget(self.chk_autosave)

    def _init_statusbar(self) -> None:
        status_bar = QStatusBar(self)
        self.setStatusBar(status_bar)

        self.lbl_status_message = QLabel("Ready")
        self.lbl_status_coords = QLabel("X: 0, Y: 0 (norm: 0, 0)")
        self.lbl_status_img_dim = QLabel("Img: -")
        self.lbl_status_file = QLabel("No file loaded")

        status_bar.addWidget(self.lbl_status_message, 1)
        status_bar.addPermanentWidget(self.lbl_status_file)
        status_bar.addPermanentWidget(self.lbl_status_img_dim)
        status_bar.addPermanentWidget(self.lbl_status_coords)

    def _init_sidebar(self) -> None:
        # Main splitter: Left = Canvas, Right = Sidebar
        main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        main_splitter.addWidget(self.view)

        sidebar = QWidget()
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(8, 8, 8, 8)
        sidebar_layout.setSpacing(10)

        # Tabs: Annotations, Samples, Prompt / Details
        self.tabs = QTabWidget()

        # Tab 1: Annotations
        tab_annos = QWidget()
        annos_layout = QVBoxLayout(tab_annos)
        annos_layout.setContentsMargins(6, 8, 6, 6)
        annos_layout.setSpacing(8)

        # BBox List
        self.list_bboxes = QListWidget()
        self.list_bboxes.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.list_bboxes.currentRowChanged.connect(self._on_list_bbox_selected)
        annos_layout.addWidget(QLabel("<b>Bounding Boxes (0-1000)</b>"))
        annos_layout.addWidget(self.list_bboxes, 1)

        # Selected Box Editor Group
        self.grp_box_edit = QGroupBox("Selected Box Properties")
        edit_layout = QVBoxLayout(self.grp_box_edit)
        edit_layout.setSpacing(6)

        # Label Row
        row_label = QHBoxLayout()
        row_label.addWidget(QLabel("Label:"))
        self.edit_box_label = QLineEdit()
        self.edit_box_label.textChanged.connect(self._on_selected_label_edited)
        row_label.addWidget(self.edit_box_label)
        edit_layout.addLayout(row_label)

        # Coordinates Spinboxes (x1, y1, x2, y2)
        grid_coords = QHBoxLayout()
        grid_coords.addWidget(QLabel("X1:"))
        self.spin_x1 = QSpinBox()
        self.spin_x1.setRange(0, 1000)
        self.spin_x1.valueChanged.connect(self._on_spin_coords_changed)
        grid_coords.addWidget(self.spin_x1)

        grid_coords.addWidget(QLabel("Y1:"))
        self.spin_y1 = QSpinBox()
        self.spin_y1.setRange(0, 1000)
        self.spin_y1.valueChanged.connect(self._on_spin_coords_changed)
        grid_coords.addWidget(self.spin_y1)
        edit_layout.addLayout(grid_coords)

        grid_coords2 = QHBoxLayout()
        grid_coords2.addWidget(QLabel("X2:"))
        self.spin_x2 = QSpinBox()
        self.spin_x2.setRange(0, 1000)
        self.spin_x2.valueChanged.connect(self._on_spin_coords_changed)
        grid_coords2.addWidget(self.spin_x2)

        grid_coords2.addWidget(QLabel("Y2:"))
        self.spin_y2 = QSpinBox()
        self.spin_y2.setRange(0, 1000)
        self.spin_y2.valueChanged.connect(self._on_spin_coords_changed)
        grid_coords2.addWidget(self.spin_y2)
        edit_layout.addLayout(grid_coords2)

        # Action Buttons
        row_box_btns = QHBoxLayout()
        self.btn_del_box = QPushButton("Delete Box (D)")
        self.btn_del_box.setObjectName("dangerBtn")
        self.btn_del_box.clicked.connect(self.delete_selected_bbox)
        row_box_btns.addWidget(self.btn_del_box)

        self.btn_dup_box = QPushButton("Duplicate (Ctrl+D)")
        self.btn_dup_box.clicked.connect(self.duplicate_selected_bbox)
        row_box_btns.addWidget(self.btn_dup_box)
        edit_layout.addLayout(row_box_btns)

        annos_layout.addWidget(self.grp_box_edit)

        # Global item buttons
        row_global_btns = QHBoxLayout()
        self.btn_add_box = QPushButton("+ Add Box (C)")
        self.btn_add_box.setCheckable(True)
        self.btn_add_box.clicked.connect(self.start_add_box)
        row_global_btns.addWidget(self.btn_add_box)

        self.btn_clear_all = QPushButton("Clear All")
        self.btn_clear_all.clicked.connect(self.clear_all_bboxes)
        row_global_btns.addWidget(self.btn_clear_all)
        annos_layout.addLayout(row_global_btns)

        self.tabs.addTab(tab_annos, "Annotations")

        # Tab 2: Samples Navigator
        tab_samples = QWidget()
        samples_layout = QVBoxLayout(tab_samples)
        samples_layout.setContentsMargins(6, 8, 6, 6)

        self.list_samples = QListWidget()
        self.list_samples.currentRowChanged.connect(self.jump_to_item)
        samples_layout.addWidget(QLabel("<b>All Dataset Samples</b>"))
        samples_layout.addWidget(self.list_samples)
        self.tabs.addTab(tab_samples, "Dataset Items")

        # Tab 3: Prompt & Raw Data
        tab_details = QWidget()
        details_layout = QVBoxLayout(tab_details)
        details_layout.setContentsMargins(6, 8, 6, 6)

        details_layout.addWidget(QLabel("<b>Human Prompt</b>"))
        self.txt_prompt = QTextEdit()
        self.txt_prompt.setReadOnly(True)
        self.txt_prompt.setMaximumHeight(120)
        details_layout.addWidget(self.txt_prompt)

        details_layout.addWidget(QLabel("<b>Raw Assistant Response JSON</b>"))
        self.txt_raw_json = QTextEdit()
        self.txt_raw_json.setMaximumHeight(200)
        details_layout.addWidget(self.txt_raw_json)

        self.btn_apply_raw_json = QPushButton("Apply Raw JSON to Boxes")
        self.btn_apply_raw_json.clicked.connect(self._apply_raw_json_to_boxes)
        details_layout.addWidget(self.btn_apply_raw_json)
        details_layout.addStretch()

        self.tabs.addTab(tab_details, "Prompt & JSON")

        sidebar_layout.addWidget(self.tabs)
        main_splitter.addWidget(sidebar)

        # Set splitter proportions and initial sizes
        main_splitter.setStretchFactor(0, 4)
        main_splitter.setStretchFactor(1, 1)
        main_splitter.setSizes([980, 360])
        self.main_splitter = main_splitter

        self.setCentralWidget(main_splitter)

    def showEvent(self, event: Any) -> None:
        super().showEvent(event)
        if self._first_show:
            self._first_show = False
            QTimer.singleShot(60, self.view.fit_image_in_view)

    def keyPressEvent(self, event: Any) -> None:
        focus_w = QApplication.focusWidget()
        if isinstance(focus_w, (QLineEdit, QTextEdit)):
            super().keyPressEvent(event)
            return

        if event.key() == Qt.Key.Key_C and not event.isAutoRepeat():
            self.start_add_box()
            event.accept()
            return
        if (
            event.key() in (Qt.Key.Key_D, Qt.Key.Key_Delete, Qt.Key.Key_Backspace)
            and not event.isAutoRepeat()
        ):
            self.delete_selected_bbox()
            event.accept()
            return
        super().keyPressEvent(event)

    def _connect_signals(self) -> None:
        self.scene.bbox_selected.connect(self._on_scene_bbox_selected)
        self.scene.bbox_deselected.connect(self._on_scene_bbox_deselected)
        self.scene.bbox_geometry_changed.connect(self._on_scene_bbox_geometry_changed)
        self.scene.bbox_edit_committed.connect(self._on_bbox_edit_committed)
        self.scene.new_bbox_drawn.connect(self._on_new_bbox_drawn)
        self.view.cursor_position_changed.connect(self._on_cursor_coords_updated)

    # -----------------------------------------------------------------------
    # Dataset Loading & Saving
    # -----------------------------------------------------------------------

    def prompt_open_file(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Open Dataset JSONL",
            str(self.jsonl_path.parent if self.jsonl_path else Path.cwd()),
            "JSONL Files (*.jsonl);;All Files (*)",
        )
        if file_path:
            self.load_dataset(file_path)

    def load_dataset(self, file_path: str) -> None:
        path = Path(file_path)
        if not path.is_file():
            QMessageBox.critical(self, "Error", f"File not found:\n{file_path}")
            return

        items = []
        try:
            with open(path, encoding="utf-8") as f:
                for _line_idx, line in enumerate(f):
                    line_str = line.strip()
                    if line_str:
                        items.append(json.loads(line_str))
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to parse JSONL:\n{e}")
            return

        if not items:
            QMessageBox.warning(self, "Warning", "JSONL file is empty.")
            return

        self.jsonl_path = path
        self.items = items
        self.dirty_indices.clear()
        self.lbl_status_file.setText(f"File: {path.name}")

        # Populate samples tab
        self.list_samples.blockSignals(True)
        self.list_samples.clear()
        for idx, it in enumerate(self.items):
            imgs = it.get("images", [])
            img_name = Path(imgs[0]).name if imgs else "No image"
            self.list_samples.addItem(f"#{idx + 1}: {img_name}")
        self.list_samples.blockSignals(False)

        self.jump_to_item(0)
        self.set_status(f"Loaded {len(items)} items from {path.name}")

    def save_dataset(self) -> None:
        if not self.jsonl_path:
            self.prompt_save_as()
            return

        self._commit_current_item()

        # Backup existing file
        backup_path = self.jsonl_path.with_suffix(self.jsonl_path.suffix + ".bak")
        try:
            if self.jsonl_path.exists():
                import shutil

                shutil.copy2(self.jsonl_path, backup_path)

            with open(self.jsonl_path, "w", encoding="utf-8") as f:
                for it in self.items:
                    f.write(json.dumps(it, ensure_ascii=False) + "\n")

            self.dirty_indices.clear()
            self._update_window_title()
            self.set_status(f"Saved {len(self.items)} items to {self.jsonl_path.name}")
        except Exception as e:
            QMessageBox.critical(self, "Save Error", f"Failed to save file:\n{e}")

    def prompt_save_as(self) -> None:
        file_path, _ = QFileDialog.getSaveFileName(
            self,
            "Save Dataset JSONL As...",
            str(self.jsonl_path if self.jsonl_path else Path.cwd() / "dataset_corrected.jsonl"),
            "JSONL Files (*.jsonl);;All Files (*)",
        )
        if file_path:
            self.jsonl_path = Path(file_path)
            self.save_dataset()

    def _commit_current_item(self) -> None:
        """Serialize current on-screen bboxes back into the active item dict."""
        if not self.items or self.current_index >= len(self.items):
            return

        item = self.items[self.current_index]
        bboxes = [b_item.to_bbox() for b_item in self.bbox_items]

        # Update GPT conversation turn
        conversations = item.get("conversations", [])
        gpt_idx = None
        for i, turn in enumerate(conversations):
            if turn.get("from") in ("gpt", "assistant"):
                gpt_idx = i
                break

        serialized = serialize_bboxes_to_gpt(bboxes)
        if gpt_idx is not None:
            conversations[gpt_idx]["value"] = serialized
        else:
            conversations.append({"from": "gpt", "value": serialized})

        item["conversations"] = conversations

    # -----------------------------------------------------------------------
    # Item Navigation
    # -----------------------------------------------------------------------

    def jump_to_item(self, index: int) -> None:
        if not self.items:
            return

        if self.chk_autosave.isChecked() and self.dirty_indices:
            self._commit_current_item()

        self.current_index = max(0, min(len(self.items) - 1, index))
        self.lbl_item_counter.setText(f" {self.current_index + 1} / {len(self.items)} ")

        self.list_samples.blockSignals(True)
        self.list_samples.setCurrentRow(self.current_index)
        self.list_samples.blockSignals(False)

        # Clear undo/redo stacks for new item
        self.undo_stack.clear()
        self.redo_stack.clear()

        self._load_current_item_ui()

    def prev_item(self) -> None:
        if self.current_index > 0:
            self.jump_to_item(self.current_index - 1)

    def next_item(self) -> None:
        if self.current_index < len(self.items) - 1:
            self.jump_to_item(self.current_index + 1)

    def _load_current_item_ui(self) -> None:
        item = self.items[self.current_index]

        # 1. Resolve and load image
        images = item.get("images", [])
        pixmap = None
        resolved_img_path = None

        if images:
            raw_path = images[0]
            # Try multiple candidate paths
            candidates = [
                Path(raw_path),
                (self.jsonl_path.parent / raw_path) if self.jsonl_path else Path(raw_path),
                Path.cwd() / raw_path,
            ]
            for cand in candidates:
                if cand.is_file():
                    resolved_img_path = cand
                    break

        if resolved_img_path and resolved_img_path.is_file():
            pixmap = QPixmap(str(resolved_img_path))
            self.lbl_status_img_dim.setText(
                f"Img: {pixmap.width()}x{pixmap.height()} ({resolved_img_path.name})"
            )
        else:
            # Placeholder pixmap if image not found
            pixmap = QPixmap(800, 1000)
            pixmap.fill(QColor("#202028"))
            p = QPainter(pixmap)
            p.setPen(QPen(QColor("#EF4444"), 2))
            p.drawText(
                QRectF(0, 0, 800, 1000),
                Qt.AlignmentFlag.AlignCenter,
                f"Image not found:\n{images[0] if images else 'None'}",
            )
            p.end()
            self.lbl_status_img_dim.setText("Img: Missing")

        self.current_pixmap = pixmap
        # set_image() calls scene.clear(), which deletes all C++ items;
        # drop our Python references to the old bbox items first.
        self.bbox_items.clear()
        self.selected_bbox_item = None
        self.scene.set_image(pixmap)
        self.view.fit_image_in_view()

        # 2. Extract conversation & bboxes
        conversations = item.get("conversations", [])
        prompt_val = ""
        gpt_val = ""
        for turn in conversations:
            if turn.get("from") in ("human", "user"):
                prompt_val = turn.get("value", "")
            elif turn.get("from") in ("gpt", "assistant"):
                gpt_val = turn.get("value", "")

        self.txt_prompt.setPlainText(prompt_val)
        self.txt_raw_json.setPlainText(gpt_val)

        # Parse bboxes
        bboxes = parse_gpt_turn(gpt_val)
        self._set_bboxes(bboxes, push_undo=False)
        self._update_window_title()

    def _set_bboxes(self, bboxes: list[BBox], push_undo: bool = True) -> None:
        if push_undo:
            current_state = [b.to_bbox() for b in self.bbox_items]
            self.undo_stack.append(current_state)
            self.redo_stack.clear()

        # Remove existing items from scene
        for it in self.bbox_items:
            if isValid(it):
                self.scene.removeItem(it)
        self.bbox_items.clear()

        # Add new items
        for bbox in bboxes:
            b_item = BBoxGraphicsItem(
                bbox, self.scene.img_width, self.scene.img_height
            )
            self.scene.addItem(b_item)
            self.bbox_items.append(b_item)

        self._refresh_bbox_list()
        if self.bbox_items:
            self.bbox_items[0].setSelected(True)
        else:
            self._on_scene_bbox_deselected()

    def _refresh_bbox_list(self) -> None:
        self.list_bboxes.blockSignals(True)
        self.list_bboxes.clear()
        for idx, it in enumerate(self.bbox_items):
            b = it.to_bbox()
            color = color_for_label(b.label)
            item_widget = QListWidgetItem(
                f"#{idx + 1:02d} [{b.label}]  [{b.x1}, {b.y1}, {b.x2}, {b.y2}]"
            )
            item_widget.setForeground(QBrush(color))
            self.list_bboxes.addItem(item_widget)

        # Select corresponding row if an item is selected
        if self.selected_bbox_item in self.bbox_items:
            idx = self.bbox_items.index(self.selected_bbox_item)
            self.list_bboxes.setCurrentRow(idx)
        self.list_bboxes.blockSignals(False)

    # -----------------------------------------------------------------------
    # Editing Actions & Handlers
    # -----------------------------------------------------------------------

    def set_mode(self, mode: int) -> None:
        self.scene.mode = mode
        if mode == MODE_SELECT:
            self.btn_select_mode.setChecked(True)
            self.btn_draw_mode.setChecked(False)
            if hasattr(self, "btn_add_box"):
                self.btn_add_box.setChecked(False)
                self.btn_add_box.setText("+ Add Box (C)")
            self.view.setCursor(Qt.CursorShape.ArrowCursor)
            self.set_status("Select Mode: Click on any box to pick it. Drag handles to resize. Drag box to move. Space to pan.")
        else:
            self.btn_select_mode.setChecked(False)
            self.btn_draw_mode.setChecked(True)
            if hasattr(self, "btn_add_box"):
                self.btn_add_box.setChecked(True)
                self.btn_add_box.setText("Drawing... (Drag on Image)")
            self.view.setCursor(Qt.CursorShape.CrossCursor)
            self.set_status("Draw Mode: Drag with mouse on the image to add a new box.")

    def _on_default_label_changed(self, text: str) -> None:
        self.scene.default_label = text.strip() or "product"

    def _on_scene_bbox_selected(self, item: BBoxGraphicsItem) -> None:
        self.selected_bbox_item = item
        self.grp_box_edit.setEnabled(True)

        # Update properties in sidebar
        b = item.to_bbox()
        self.edit_box_label.blockSignals(True)
        self.edit_box_label.setText(b.label)
        self.edit_box_label.blockSignals(False)

        self.spin_x1.blockSignals(True)
        self.spin_y1.blockSignals(True)
        self.spin_x2.blockSignals(True)
        self.spin_y2.blockSignals(True)
        self.spin_x1.setValue(b.x1)
        self.spin_y1.setValue(b.y1)
        self.spin_x2.setValue(b.x2)
        self.spin_y2.setValue(b.y2)
        self.spin_x1.blockSignals(False)
        self.spin_y1.blockSignals(False)
        self.spin_x2.blockSignals(False)
        self.spin_y2.blockSignals(False)

        # Sync list widget selection and ensure it is scrolled into view
        if item in self.bbox_items:
            idx = self.bbox_items.index(item)
            self.list_bboxes.blockSignals(True)
            self.list_bboxes.setCurrentRow(idx)
            list_item = self.list_bboxes.item(idx)
            if list_item:
                self.list_bboxes.scrollToItem(list_item)
            self.list_bboxes.blockSignals(False)

    def _on_scene_bbox_deselected(self) -> None:
        self.selected_bbox_item = None
        self.grp_box_edit.setEnabled(False)
        self.list_bboxes.blockSignals(True)
        self.list_bboxes.clearSelection()
        self.list_bboxes.blockSignals(False)

    def _on_scene_bbox_geometry_changed(self, item: BBoxGraphicsItem) -> None:
        b = item.to_bbox()
        self.spin_x1.blockSignals(True)
        self.spin_y1.blockSignals(True)
        self.spin_x2.blockSignals(True)
        self.spin_y2.blockSignals(True)
        self.spin_x1.setValue(b.x1)
        self.spin_y1.setValue(b.y1)
        self.spin_x2.setValue(b.x2)
        self.spin_y2.setValue(b.y2)
        self.spin_x1.blockSignals(False)
        self.spin_y1.blockSignals(False)
        self.spin_x2.blockSignals(False)
        self.spin_y2.blockSignals(False)

        # Update text in list
        if item in self.bbox_items:
            idx = self.bbox_items.index(item)
            list_item = self.list_bboxes.item(idx)
            if list_item:
                list_item.setText(f"#{idx + 1:02d} [{b.label}]  [{b.x1}, {b.y1}, {b.x2}, {b.y2}]")

    def _on_bbox_edit_committed(self) -> None:
        self._push_undo_checkpoint()
        self._mark_dirty()

    def _on_list_bbox_selected(self, row: int) -> None:
        if 0 <= row < len(self.bbox_items):
            item = self.bbox_items[row]
            if not isValid(item):
                return
            for it in self.scene.selectedItems():
                if it != item:
                    it.setSelected(False)
            item.setSelected(True)

    def _on_selected_label_edited(self, text: str) -> None:
        if self.selected_bbox_item:
            self._push_undo_checkpoint()
            new_label = text.strip() or "product"
            self.selected_bbox_item.set_label(new_label)
            self._refresh_bbox_list()
            self._mark_dirty()

    def _on_spin_coords_changed(self) -> None:
        if self.selected_bbox_item:
            self._push_undo_checkpoint()
            x1 = self.spin_x1.value()
            y1 = self.spin_y1.value()
            x2 = self.spin_x2.value()
            y2 = self.spin_y2.value()
            self.selected_bbox_item.set_bbox_coords(x1, y1, x2, y2)
            self._refresh_bbox_list()
            self._mark_dirty()

    def start_add_box(self) -> None:
        """Switch to draw mode so user can drag mouse to add a box."""
        self.set_mode(MODE_DRAW)

    def _on_new_bbox_drawn(self, bbox: BBox) -> None:
        self._push_undo_checkpoint()
        b_item = BBoxGraphicsItem(
            bbox, self.scene.img_width, self.scene.img_height
        )
        self.scene.addItem(b_item)
        self.bbox_items.append(b_item)
        self._refresh_bbox_list()

        # Deselect any other boxes and exclusively pick the new box
        for it in self.scene.selectedItems():
            if it != b_item:
                it.setSelected(False)
        b_item.setSelected(True)
        self._mark_dirty()

        # Return to select mode so user can immediately move/adjust handles/edit
        self.set_mode(MODE_SELECT)
        self.set_status(f"Added box: {bbox.label} [{bbox.x1}, {bbox.y1}, {bbox.x2}, {bbox.y2}]. Picked for editing.")

    def add_new_default_bbox(self, bbox: BBox | None = None) -> None:
        """Programmatically add a bounding box (or default) and pick it."""
        if bbox is None:
            bbox = BBox(250, 250, 750, 750, self.edit_default_label.text().strip() or "product")
        self._on_new_bbox_drawn(bbox)

    def delete_selected_bbox(self) -> None:
        target = self.selected_bbox_item
        if (not target or target not in self.bbox_items) and self.bbox_items:
            cur_row = self.list_bboxes.currentRow()
            if 0 <= cur_row < len(self.bbox_items):
                target = self.bbox_items[cur_row]
            else:
                target = self.bbox_items[-1]

        if target and target in self.bbox_items:
            self._push_undo_checkpoint()
            idx = self.bbox_items.index(target)
            self.scene.removeItem(target)
            self.bbox_items.remove(target)
            self.selected_bbox_item = None
            self._refresh_bbox_list()
            if self.bbox_items:
                next_idx = min(idx, len(self.bbox_items) - 1)
                self.bbox_items[next_idx].setSelected(True)
            else:
                self._on_scene_bbox_deselected()
            self._mark_dirty()
            self.set_status("Deleted selected box.")

    def duplicate_selected_bbox(self) -> None:
        target = self.selected_bbox_item
        if (not target or target not in self.bbox_items) and self.bbox_items:
            cur_row = self.list_bboxes.currentRow()
            if 0 <= cur_row < len(self.bbox_items):
                target = self.bbox_items[cur_row]
            else:
                target = self.bbox_items[-1]

        if target:
            self._push_undo_checkpoint()
            b = target.to_bbox()
            # Offset slightly
            dup_box = BBox(
                min(1000, b.x1 + 25),
                min(1000, b.y1 + 25),
                min(1000, b.x2 + 25),
                min(1000, b.y2 + 25),
                b.label,
            ).normalized()
            b_item = BBoxGraphicsItem(
                dup_box, self.scene.img_width, self.scene.img_height
            )
            self.scene.addItem(b_item)
            self.bbox_items.append(b_item)
            self._refresh_bbox_list()
            b_item.setSelected(True)
            self._mark_dirty()

    def clear_all_bboxes(self) -> None:
        if not self.bbox_items:
            return
        ans = QMessageBox.question(
            self,
            "Clear All Boxes",
            "Are you sure you want to remove all bounding boxes for this sample?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        )
        if ans == QMessageBox.StandardButton.Yes:
            self._push_undo_checkpoint()
            for it in self.bbox_items:
                self.scene.removeItem(it)
            self.bbox_items.clear()
            self._on_scene_bbox_deselected()
            self._refresh_bbox_list()
            self._mark_dirty()

    def _apply_raw_json_to_boxes(self) -> None:
        raw_text = self.txt_raw_json.toPlainText().strip()
        bboxes = parse_gpt_turn(raw_text)
        if not bboxes:
            QMessageBox.warning(self, "Invalid JSON", "Could not parse any bounding boxes from the text.")
            return
        self._set_bboxes(bboxes, push_undo=True)
        self._mark_dirty()
        self.set_status(f"Applied {len(bboxes)} boxes from raw JSON.")

    # -----------------------------------------------------------------------
    # Undo / Redo & Status Helpers
    # -----------------------------------------------------------------------

    def _push_undo_checkpoint(self) -> None:
        current_state = [b.to_bbox() for b in self.bbox_items]
        self.undo_stack.append(current_state)
        self.redo_stack.clear()

    def undo(self) -> None:
        if not self.undo_stack:
            return
        current_state = [b.to_bbox() for b in self.bbox_items]
        self.redo_stack.append(current_state)
        prev_state = self.undo_stack.pop()

        for it in self.bbox_items:
            self.scene.removeItem(it)
        self.bbox_items.clear()

        for b in prev_state:
            b_item = BBoxGraphicsItem(b, self.scene.img_width, self.scene.img_height)
            self.scene.addItem(b_item)
            self.bbox_items.append(b_item)

        self._refresh_bbox_list()
        if self.bbox_items:
            self.bbox_items[-1].setSelected(True)
        else:
            self._on_scene_bbox_deselected()
        self._mark_dirty()
        self.set_status("Undo performed.")

    def redo(self) -> None:
        if not self.redo_stack:
            return
        current_state = [b.to_bbox() for b in self.bbox_items]
        self.undo_stack.append(current_state)
        next_state = self.redo_stack.pop()

        for it in self.bbox_items:
            self.scene.removeItem(it)
        self.bbox_items.clear()

        for b in next_state:
            b_item = BBoxGraphicsItem(b, self.scene.img_width, self.scene.img_height)
            self.scene.addItem(b_item)
            self.bbox_items.append(b_item)

        self._refresh_bbox_list()
        if self.bbox_items:
            self.bbox_items[-1].setSelected(True)
        else:
            self._on_scene_bbox_deselected()
        self._mark_dirty()
        self.set_status("Redo performed.")

    def _mark_dirty(self) -> None:
        self.dirty_indices.add(self.current_index)
        self._update_window_title()

    def _update_window_title(self) -> None:
        title = "VLM Dataset Corrector"
        if self.jsonl_path:
            title += f" - {self.jsonl_path.name}"
        if self.dirty_indices:
            title += " *"
        self.setWindowTitle(title)

    def set_status(self, text: str) -> None:
        self.lbl_status_message.setText(text)

    def _on_cursor_coords_updated(self, px_x: int, px_y: int, norm_x: int, norm_y: int) -> None:
        self.lbl_status_coords.setText(
            f"Cursor: ({px_x}, {px_y})px | Norm: ({norm_x}, {norm_y})/1000"
        )


# ---------------------------------------------------------------------------
# CLI Entry Point
# ---------------------------------------------------------------------------

def main() -> None:
    import sys

    app = QApplication(sys.argv)
    app.setApplicationName("VLM Dataset Corrector")

    initial_path = sys.argv[1] if len(sys.argv) > 1 else None
    window = DatasetCorrector(initial_path)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
