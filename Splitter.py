import os
import sys
import threading
import tkinter as tk
from tkinter import ttk, messagebox, filedialog, simpledialog
import json
import fitz  # PyMuPDF
import cv2
from PIL import Image, ImageTk
import numpy as np
import re
import easyocr
from datetime import datetime
import shutil
import requests

# ---------------------------- VERSION ----------------------------
VERSION_FILE = "version.json"
try:
    with open(VERSION_FILE, "r", encoding="utf-8") as f:
        __version__ = json.load(f).get("version", "0.0.0")
except Exception:
    __version__ = "0.0.0"

GITHUB_VERSION_JSON = "https://raw.githubusercontent.com/GermoEis/Splitter/main/version.json"

# ---------------------------- CONFIG ----------------------------
CONFIG_FILE = r"Z:/share/Ocr_Splittija/config.json"

try:
    ocr_engine = easyocr.Reader(["ru"], gpu=True)
except Exception:
    ocr_engine = easyocr.Reader(["ru"], gpu=False)

def load_config():
    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, "r", encoding="utf-8") as f:
                cfg = json.load(f)
        except Exception:
            cfg = {}
    else:
        cfg = {}
    if "jobs_conditions" not in cfg and "jobs" in cfg:
        jobs_old = cfg.get("jobs", {})
        jc = {}
        for mainjob, subs in jobs_old.items():
            jc[mainjob] = {}
            for sub in subs:
                jc[mainjob][sub] = {"include": [], "exclude": [], "all_must_match": False}
        cfg["jobs_conditions"] = jc
    cfg.setdefault("jobs_conditions", {})
    cfg.setdefault("poppler_path", "")
    return cfg

def save_config(cfg):
    os.makedirs(os.path.dirname(CONFIG_FILE), exist_ok=True)
    with open(CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)

# ---------------------------- OCR ----------------------------
def ocr_header(image, header_ratio=0.15):
    h = image.shape[0]
    crop_h = max(1, int(h * header_ratio))
    header_img = image[:crop_h, :].copy()
    try:
        result = ocr_engine.readtext(header_img)
        text = " ".join([line[1] for line in result]) if result else ""
    except Exception:
        text = ""
    return text, header_img

def correct_orientation(image):
    return image

def extract_text_with_ocr(page):
    pix = page.get_pixmap()
    mode = "RGB" if pix.n < 4 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples).convert("RGB")
    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
    img_cv = correct_orientation(img_cv)
    try:
        result = ocr_engine.readtext(img_cv)
        text = " ".join([line[1] for line in result]) if result else ""
    except Exception:
        text = ""
    return text

def check_include(text, inc_list, all_must_match):
    if not inc_list:
        return False
    matches = []
    for entry in inc_list:
        if isinstance(entry, str):
            matches.append(entry.casefold() in text.casefold())
        elif isinstance(entry, list):
            matches.append(any(word.casefold() in text.casefold() for word in entry))
    return all(matches) if all_must_match else any(matches)



def split_pdfs(pdf_paths, conds_dict, output_dir, progress_callback=None):
    all_results = {}
    for pdf_path in pdf_paths:
        results = {}
        doc = fitz.open(pdf_path)
        total_pages = doc.page_count

        fallback_sj = None
        for sj, conds in conds_dict.items():
            if not conds.get("include") and not conds.get("exclude"):
                fallback_sj = sj
                break

        current_sj = None
        current_batch = []
        current_keywords = None

        def save_batch(batch_sj, batch_pages, keywords):
            if not batch_pages:
                return
            safe_sj = re.sub(r'[<>:"/\\|?*]+', "_", batch_sj or "unknown_job")
            sj_dir = os.path.join(output_dir, safe_sj)
            os.makedirs(sj_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(pdf_path))[0]
            safe_base_name = re.sub(r'[<>:"/\\|?* ]+', "_", base_name)
            if len(batch_pages) == 1:
                out_path = os.path.join(sj_dir, f"{safe_base_name}_page_{batch_pages[0]+1}.pdf")
            else:
                out_path = os.path.join(sj_dir, f"{safe_base_name}_pages_{batch_pages[0]+1}_{batch_pages[-1]+1}.pdf")
            writer = fitz.open()
            for p in batch_pages:
                writer.insert_pdf(doc, from_page=p, to_page=p)
            writer.save(out_path)
            writer.close()
            results.setdefault(batch_sj, []).append({"path": out_path, "found_keywords": keywords})

        for i in range(total_pages):
            page = doc[i]
            text = page.get_text("text")
            if not text.strip():
                text = extract_text_with_ocr(page)

            matched_sj = None
            matched_keywords = None

            # kontrollime kõiki reegleid
            for sj, conds in conds_dict.items():
                inc = conds.get("include", [])
                exc = conds.get("exclude", [])
                all_must_match = conds.get("all_must_match", False)

                if inc or exc:
                    inc_ok = check_include(text, inc, all_must_match)
                    exc_ok = not any((word.casefold() in text.casefold()) for word in exc) if exc else True
                    if inc_ok and exc_ok:
                        matched_sj = sj
                        found_inc = []
                        for entry in inc:
                            if isinstance(entry, str) and entry.casefold() in text.casefold():
                                found_inc.append(entry)
                            elif isinstance(entry, list):
                                hits = [w for w in entry if w.casefold() in text.casefold()]
                                if hits:
                                    found_inc.append(hits)
                        matched_keywords = {
                            "include": found_inc,
                            "exclude": [word for word in exc if word.casefold() in text.casefold()],
                        }
                        break

            if matched_sj:  
                # Kui uus reegel aktiveerus → salvesta eelmine batch ja alusta uut
                if current_batch:
                    save_batch(current_sj, current_batch, current_keywords)
                    current_batch = []
                current_sj = matched_sj
                current_keywords = matched_keywords
            elif current_sj is None:  
                # kui mitte midagi pole veel alustatud
                current_sj = fallback_sj
                current_keywords = None

            # Lisa leht jooksvale batchile
            current_batch.append(i)

            if progress_callback:
                try:
                    progress_callback(i + 1, total_pages)
                except Exception:
                    pass

        # Lõpus salvesta viimased lehed
        if current_sj and current_batch:
            save_batch(current_sj, current_batch, current_keywords)

        doc.close()
        all_results[os.path.basename(pdf_path)] = results

    return all_results




# ---------------------------- MODERN UI ----------------------------
class ModernPDFSplitter:
    def __init__(self, root):
        self.root = root
        root.title("🗂 Modern PDF Splitter (EasyOCR)")
        root.geometry("1300x850")
        root.configure(bg="#f7f9fc")

        self.config = load_config()
        self.jobs = self.config.get("jobs_conditions", {})

        self.pdf_list = []
        self.current_pdf_index = 0
        self.preview_images = []
        self.current_preview_index = 0

        self.output_dir = None
        self.processed_dir = None

        self.setup_ui()
        self.refresh_tree()
        threading.Thread(target=self.check_for_update_background, daemon=True).start()

    def setup_ui(self):
        style = ttk.Style(self.root)
        style.theme_use('clam')
        style.configure("Treeview", background="#ffffff", fieldbackground="#ffffff", rowheight=25)
        style.configure("TButton", font=("Arial", 10), padding=6)

        # Top Frame
        top_frame = tk.Frame(self.root, bg="#f7f9fc")
        top_frame.pack(fill=tk.X, padx=10, pady=10)
        btn_bg = "#4a90e2"
        btn_fg = "white"
        tk.Button(top_frame, text="Vali Poppler asukoht", bg=btn_bg, fg=btn_fg, relief=tk.FLAT, command=self.set_poppler_path).pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="Vali väljundkaust", bg=btn_bg, fg=btn_fg, relief=tk.FLAT, command=self.set_output_dir).pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="Vali õige split kaust", bg=btn_bg, fg=btn_fg, relief=tk.FLAT, command=self.set_processed_dir).pack(side=tk.LEFT, padx=4)
        tk.Button(top_frame, text="Vali PDF-id", bg=btn_bg, fg=btn_fg, relief=tk.FLAT, command=self.load_pdfs).pack(side=tk.LEFT, padx=4)
        self.label_output = tk.Label(top_frame, text="Output folder: [not set]", bg="#f7f9fc")
        self.label_output.pack(side=tk.LEFT, padx=12)

        # Main PanedWindow
        main_pane = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # Left: Jobs Tree
        self.tree_frame = tk.Frame(main_pane, bd=2, relief=tk.GROOVE, bg="white")
        tk.Label(self.tree_frame, text="Jobs Hierarchy", bg="white", font=("Arial", 12, "bold")).pack(anchor="w", padx=6, pady=6)
        self.tree = ttk.Treeview(self.tree_frame)
        self.tree.pack(fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.tree.bind("<<TreeviewSelect>>", self.on_tree_select)

        tree_btn_frame = tk.Frame(self.tree_frame, bg="white")
        tree_btn_frame.pack(fill=tk.X, padx=6, pady=6)
        tk.Button(tree_btn_frame, text="Lisa Main Job", bg="#34a853", fg="white", relief=tk.FLAT, command=self.add_main_job).pack(side=tk.LEFT)
        tk.Button(tree_btn_frame, text="Lisa Subjob", bg="#fbbc05", fg="white", relief=tk.FLAT, command=self.add_subjob).pack(side=tk.LEFT, padx=4)
        tk.Button(tree_btn_frame, text="Muuda", bg="#f37321", fg="white", relief=tk.FLAT, command=self.edit_tree_item).pack(side=tk.LEFT, padx=4)
        tk.Button(tree_btn_frame, text="Kustuta", bg="#ea4335", fg="white", relief=tk.FLAT, command=self.delete_tree_item).pack(side=tk.LEFT, padx=4)

        # Right: Details + Preview
        self.detail_frame = tk.Frame(main_pane, bd=2, relief=tk.GROOVE, bg="white")
        conds_top = tk.Frame(self.detail_frame, bg="white")
        conds_top.pack(fill=tk.X, padx=6, pady=6)

        # Include & Exclude
        left_col = tk.Frame(conds_top, bg="white")
        left_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0,6))
        tk.Label(left_col, text="Sisaldab sõnu", bg="white").pack(anchor="w")
        self.list_include = tk.Listbox(left_col, height=6)
        self.list_include.pack(fill=tk.BOTH, expand=True)
        btn_inc = tk.Frame(left_col, bg="white")
        btn_inc.pack(fill=tk.X, pady=4)
        tk.Button(btn_inc, text="Lisa", bg="#34a853", fg="white", relief=tk.FLAT, command=lambda: self.add_condition("include")).pack(side=tk.LEFT)
        tk.Button(btn_inc, text="Muuda", bg="#fbbc05", fg="white", relief=tk.FLAT, command=lambda: self.edit_condition("include")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_inc, text="Kustuta", bg="#ea4335", fg="white", relief=tk.FLAT, command=lambda: self.del_condition("include")).pack(side=tk.LEFT, padx=4)

        right_col = tk.Frame(conds_top, bg="white")
        right_col.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        tk.Label(right_col, text="Ei sisalda sõnu", bg="white").pack(anchor="w")
        self.list_exclude = tk.Listbox(right_col, height=6)
        self.list_exclude.pack(fill=tk.BOTH, expand=True)
        btn_exc = tk.Frame(right_col, bg="white")
        btn_exc.pack(fill=tk.X, pady=4)
        tk.Button(btn_exc, text="Lisa", bg="#34a853", fg="white", relief=tk.FLAT, command=lambda: self.add_condition("exclude")).pack(side=tk.LEFT)
        tk.Button(btn_exc, text="Muuda", bg="#fbbc05", fg="white", relief=tk.FLAT, command=lambda: self.edit_condition("exclude")).pack(side=tk.LEFT, padx=4)
        tk.Button(btn_exc, text="Kustuta", bg="#ea4335", fg="white", relief=tk.FLAT, command=lambda: self.del_condition("exclude")).pack(side=tk.LEFT, padx=4)

        self.all_match_var = tk.BooleanVar(value=False)
        tk.Checkbutton(self.detail_frame, text="Kõik sisaldatavad sõnad peavad olema", variable=self.all_match_var, bg="white", command=self.toggle_all_must_match).pack(anchor="w", padx=8, pady=(0,6))

        # Preview canvas
        self.canvas_label = tk.Label(self.detail_frame, text="PDF eelvaade", bg="white")
        self.canvas_label.pack(anchor="w", padx=8)
        self.canvas = tk.Canvas(self.detail_frame, bg="#e8eaf6", height=250)
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=8, pady=(0,6))

        nav_frame = tk.Frame(self.detail_frame, bg="white")
        nav_frame.pack(fill=tk.X, padx=8, pady=(0,8))
        tk.Button(nav_frame, text="Eelmine leht", bg="#4a90e2", fg="white", relief=tk.FLAT, command=lambda: self.show_preview(-1)).pack(side=tk.LEFT)
        tk.Button(nav_frame, text="Järgmine leht", bg="#4a90e2", fg="white", relief=tk.FLAT, command=lambda: self.show_preview(1)).pack(side=tk.LEFT, padx=6)
        tk.Button(nav_frame, text="Alusta Splittimist", bg="#0f9d58", fg="white", relief=tk.FLAT, command=self.start_split).pack(side=tk.RIGHT)
        self.page_label = tk.Label(nav_frame, text="Page: 0/0", bg="white")
        self.page_label.pack(side=tk.LEFT, padx=8)

        main_pane.add(self.tree_frame, weight=1)
        main_pane.add(self.detail_frame, weight=3)

        # Version label
        self.version_label = tk.Label(self.root, text=f"Version: {__version__}", bg="#f7f9fc", font=("Arial", 10, "italic"))
        self.version_label.pack(anchor="se", side=tk.BOTTOM, padx=10, pady=4)

       # threading.Thread(target=self.check_for_update_background, daemon=True).start()



    # ---------------- Versioon & Uuendus ----------------
    def check_for_update_background(self):
        try:
            r = requests.get(GITHUB_VERSION_JSON, timeout=5)
            if r.status_code == 200:
                data = r.json()
                latest_version = data.get("version")
                download_url = data.get("download_url")  
                if latest_version and self.version_compare(latest_version, __version__):
                    self.root.after(0, self.prompt_update, latest_version, download_url)
        except Exception:
            pass

    def prompt_update(self, latest_version, download_url=None):
        if download_url is None:
           # messagebox.showwarning("Uuendus", "Download URL puudub version.json failist.")
            return

        if messagebox.askyesno(
            "Uus versioon saadaval",
            f"Saadaval on uus versioon {latest_version}.\nUuendada jooksvalt rakenduse skript?"
        ):
            self.download_and_replace(download_url)


    def download_and_replace(self, download_url):
        try:
            # Lae alla skript
            r = requests.get(download_url, timeout=10)
            r.raise_for_status()
            script_path = os.path.realpath(sys.argv[0])
            backup_path = script_path + ".bak"
            shutil.copyfile(script_path, backup_path)
            with open(script_path, "wb") as f:
                f.write(r.content)

            # Uuenda jooksvalt versiooni UI-s
            data = requests.get(GITHUB_VERSION_JSON, timeout=5).json()
            new_version = data.get("version", "0.0.0")
            global __version__
            __version__ = new_version
            self.version_label.config(text=f"Versioon: {__version__}")

            # ❌ Uus osa: salvesta kohalik version.json
            with open(VERSION_FILE, "w", encoding="utf-8") as vf:
                json.dump({"version": __version__}, vf, indent=2, ensure_ascii=False)

            messagebox.showinfo(
                "Uuendus tehtud",
                f"Rakendus on uuendatud.\nBackup tehtud: {backup_path}\nTaaskäivita rakendus."
            )
        except Exception as e:
            messagebox.showerror("Viga", f"Uuenduse allalaadimine ebaõnnestus: {e}")


    def version_compare(self, latest, current):
        def parse(v):
            parts = v.split(".")
            nums = []
            for p in parts:
                try:
                    nums.append(int(p))
                except ValueError:
                    nums.append(0)
            return nums
        return parse(latest) > parse(current)


    # ---------------- Jobs / Conditions ----------------
    def refresh_tree(self):
        self.tree.delete(*self.tree.get_children())
        for mainjob, subs in self.jobs.items():
            parent = self.tree.insert("", "end", text=mainjob, open=True)
            # subs on dict (subjob -> conds) — iteratsioon annab subjob nimed
            for subjob in subs.keys():
                self.tree.insert(parent, "end", text=subjob)

    def refresh_conditions_view(self):
        self.list_include.delete(0, tk.END)
        self.list_exclude.delete(0, tk.END)
        if self.current_subjob:
            mainjob, subjob = self.current_subjob
            conds = self.jobs.get(mainjob, {}).get(subjob, {"include": [], "exclude": [], "all_must_match": False})
            for w in conds.get("include", []):
                self.list_include.insert(tk.END, w)
            for w in conds.get("exclude", []):
                self.list_exclude.insert(tk.END, w)
            self.all_match_var.set(conds.get("all_must_match", False))
        elif getattr(self, "current_mainjob", None):
            self.list_include.insert(tk.END, f"[MainJob '{self.current_mainjob}' — vali Subjob, et näha reegleid]")
            self.all_match_var.set(False)

    def on_tree_select(self, event):
        sel = self.tree.selection()
        if not sel:
            return
        item = sel[0]
        parent = self.tree.parent(item)
        if parent:
            mainjob = self.tree.item(parent, "text")
            subjob = self.tree.item(item, "text")
            self.current_mainjob = mainjob
            self.current_subjob = (mainjob, subjob)
        else:
            self.current_mainjob = self.tree.item(item, "text")
            self.current_subjob = None
        self.refresh_conditions_view()

    def add_main_job(self):
        name = simpledialog.askstring("Main Job", "Sisesta Main Job nimi:")
        if not name:
            return
        if name in self.jobs:
            messagebox.showerror("Error", "Main Job juba olemas")
            return
        self.jobs[name] = {}
        self.config["jobs_conditions"] = self.jobs
        save_config(self.config)
        self.refresh_tree()

    def add_subjob(self):
        sel = self.tree.selection()
        if not sel:
            messagebox.showerror("Error", "Vali Main Job, kuhu lisada Subjob")
            return
        item = sel[0]
        parent = self.tree.parent(item)
        if parent:
            main_item = parent
        else:
            main_item = item
        mainjob = self.tree.item(main_item, "text")
        name = simpledialog.askstring("Subjob", f"Sisesta Subjob nimi (Main: {mainjob}):")
        if not name:
            return
        if name in self.jobs.get(mainjob, {}):
            messagebox.showerror("Error", "Subjob juba olemas selles MainJob'is")
            return
        self.jobs.setdefault(mainjob, {})[name] = {"include": [], "exclude": [], "all_must_match": False}
        self.config["jobs_conditions"] = self.jobs
        save_config(self.config)
        self.refresh_tree()

    def edit_tree_item(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = sel[0]
        parent = self.tree.parent(item)
        old_name = self.tree.item(item, "text")
        new_name = simpledialog.askstring("Edit", "Uus nimi:", initialvalue=old_name)
        if not new_name or new_name == old_name:
            return
        if parent:  # renaming subjob
            mainjob = self.tree.item(parent, "text")
            subs = self.jobs.get(mainjob, {})
            if new_name in subs:
                messagebox.showerror("Error", "Seda subjob nime juba kasutatakse selles mainjob'is")
                return
            conds = subs.pop(old_name, {})
            subs[new_name] = conds
        else:  # renaming mainjob
            if new_name in self.jobs:
                messagebox.showerror("Error", "Main Job nimega juba eksisteerib")
                return
            conds = self.jobs.pop(old_name, {})
            self.jobs[new_name] = conds
        self.config["jobs_conditions"] = self.jobs
        save_config(self.config)
        self.refresh_tree()

    def delete_tree_item(self):
        sel = self.tree.selection()
        if not sel:
            return
        item = sel[0]
        parent = self.tree.parent(item)
        name = self.tree.item(item, "text")
        if parent:
            mainjob = self.tree.item(parent, "text")
            if not messagebox.askyesno("Kustuta", f"Kustutada Subjob '{name}' MainJob'ist '{mainjob}'?"):
                return
            self.jobs.get(mainjob, {}).pop(name, None)
        else:
            if not messagebox.askyesno("Kustuta", f"Kustutada MainJob '{name}' koos kõikide subjobidega?"):
                return
            self.jobs.pop(name, None)
        self.config["jobs_conditions"] = self.jobs
        save_config(self.config)
        self.refresh_tree()
        self.current_mainjob = None
        self.current_subjob = None
        self.refresh_conditions_view()

    # Conditions manipulation
    def toggle_all_must_match(self):
        if not self.current_subjob:
            return
        mainjob, subjob = self.current_subjob
        self.jobs.setdefault(mainjob, {}).setdefault(subjob, {}).setdefault("all_must_match", False)
        self.jobs[mainjob][subjob]["all_must_match"] = bool(self.all_match_var.get())
        self.config["jobs_conditions"] = self.jobs
        save_config(self.config)

    def add_condition(self, list_type):
        if not self.current_subjob:
            messagebox.showerror("Error", "Vali esmalt Subjob (puust)")
            return
        word = simpledialog.askstring("Lisa", f"Sisesta sõna ({list_type}):")
        if not word:
            return
        mainjob, subjob = self.current_subjob
        self.jobs.setdefault(mainjob, {}).setdefault(subjob, {}).setdefault(list_type, []).append(word)
        self.config["jobs_conditions"] = self.jobs
        save_config(self.config)
        self.refresh_conditions_view()

    def edit_condition(self, list_type):
        if not self.current_subjob:
            return
        lb = self.list_include if list_type == "include" else self.list_exclude
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        old = lb.get(idx)
        new = simpledialog.askstring("Muuda sõna", "Uus sõna:", initialvalue=old)
        if not new:
            return
        mainjob, subjob = self.current_subjob
        self.jobs[mainjob][subjob][list_type][idx] = new
        self.config["jobs_conditions"] = self.jobs
        save_config(self.config)
        self.refresh_conditions_view()

    def del_condition(self, list_type):
        if not self.current_subjob:
            return
        lb = self.list_include if list_type == "include" else self.list_exclude
        sel = lb.curselection()
        if not sel:
            return
        idx = sel[0]
        mainjob, subjob = self.current_subjob
        self.jobs[mainjob][subjob][list_type].pop(idx)
        self.config["jobs_conditions"] = self.jobs
        save_config(self.config)
        self.refresh_conditions_view()

    # ---------------- Preview / PDF loading ----------------
    def load_pdfs(self):
        paths = filedialog.askopenfilenames(filetypes=[("PDF Files", "*.pdf")])
        if not paths:
            return
        self.pdf_list = list(paths)
        self.current_pdf_index = 0
        self.current_pdf = self.pdf_list[self.current_pdf_index]
        try:
            doc = fitz.open(self.current_pdf)
            self.preview_images = []
            max_preview = min(8, doc.page_count)
            for i in range(max_preview):
                pix = doc[i].get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
                mode = "RGB" if pix.n < 4 else "RGBA"
                img = Image.frombytes(mode, [pix.width, pix.height], pix.samples).convert("RGB")
                self.preview_images.append(img)
            doc.close()
            self.current_preview_index = 0
            self.show_preview(0)
        except Exception as e:
            messagebox.showerror("Viga", f"Eelvaate laadimine ebaõnnestus: {e}")

    def show_preview(self, delta):
        if not self.preview_images:
            self.canvas.delete("all")
            self.page_label.config(text="Leht: 0/0")
            return
        self.current_preview_index = max(0, min(self.current_preview_index + delta, len(self.preview_images) - 1))
        pil = self.preview_images[self.current_preview_index]
        img_cv = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
        _, header_img = ocr_header(img_cv, header_ratio=0.20)
        header_rgb = cv2.cvtColor(header_img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(header_rgb)
        canvas_w = max(200, self.canvas.winfo_width())
        img_pil.thumbnail((canvas_w, 200))
        self.tk_img = ImageTk.PhotoImage(img_pil)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_img)
        self.page_label.config(text=f"Leht: {self.current_preview_index+1}/{len(self.preview_images)}")

    # ---------------- Settings ----------------
    def set_poppler_path(self):
        path = filedialog.askdirectory(title="Poppler bin path (valikuline)")
        if path:
            self.config["poppler_path"] = path
            save_config(self.config)
            messagebox.showinfo("OK", "Poppler path salvestatud (valikuline)")

    def set_output_dir(self):
        dir_path = filedialog.askdirectory(title="Väljundkaust")
        if dir_path:
            self.output_dir = dir_path
            self.label_output.config(text=f"Väljundkaust: {dir_path}")

    def set_processed_dir(self):
        dir_path = filedialog.askdirectory(title="Processed kaust (kuhu algsed failid tõstetakse)")
        if dir_path:
            self.processed_dir = dir_path
            messagebox.showinfo("OK", f"Processed kaust määratud: {dir_path}")

    # ---------------- Splitting workflow ----------------
    def start_split(self):
        if not self.pdf_list:
            messagebox.showerror("Error", "Vali vähemalt üks PDF (Vali PDF(id))")
            return
        if not getattr(self, "current_mainjob", None):
            messagebox.showerror("Error", "Vali Main Job (puust, mitte Subjob)!")
            return
        mainjob = self.current_mainjob
        conds_dict = self.jobs.get(mainjob, {})
        if not conds_dict:
            if not messagebox.askyesno("Hoiatus", "Valitud Main Job'il puuduvad subjob'id või reeglid — jätkata ikkagi?"):
                return
        if not self.output_dir:
            self.set_output_dir()
            if not self.output_dir:
                return

        out_dir = self.output_dir
        pdfs_to_process = list(self.pdf_list)

        # progress aken
        self.progress_win = tk.Toplevel(self.root)
        self.progress_win.title("Töötlen...")
        self.progress_label = tk.Label(self.progress_win, text="Alustamas...")
        self.progress_label.pack(padx=10, pady=6)
        self.progressbar = ttk.Progressbar(self.progress_win, length=450, mode="determinate")
        self.progressbar.pack(padx=10, pady=(0, 10))
        self.progressbar["value"] = 0
        tk.Button(self.progress_win, text="Sulge", command=self.progress_win.destroy).pack(pady=(0, 8))

        def _update_ui_progress(page, total, percent):
            self.progress_label.config(text=f"Töötlen: leht {page}/{total}")
            self.progressbar["value"] = percent
            self.progressbar.update_idletasks()

        def threaded_split():
            total_pdfs = len(pdfs_to_process)
            all_summary = {}
            try:
                for pdf_idx, pdf_path in enumerate(pdfs_to_process, start=1):
                    def progress_callback(page, total):
                        overall_progress = int(((pdf_idx - 1 + (page / total)) / total_pdfs) * 100)
                        self.root.after(0, _update_ui_progress, page, total, overall_progress)

                    results = split_pdfs([pdf_path], conds_dict, out_dir, progress_callback=progress_callback)
                    # salvesta log
                    base = os.path.basename(pdf_path)
                    # log_name = f"{base}_split_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    # log_path = os.path.join(out_dir, log_name)
                    # with open(log_path, "w", encoding="utf-8") as lf:
                    #     lf.write(f"Split log for: {pdf_path}\n")
                    #     for sj, pages in results.get(base, {}).items():
                    #         lf.write(f"Subjob '{sj}': {len(pages)} faili\n")
                    #         for page_info in pages:
                    #             lf.write(f"  {page_info['path']}\n")
                    #             fk = page_info.get("found_keywords", {})
                    #             lf.write(f"    Leitud sõnad: include={fk.get('include')}, exclude={fk.get('exclude')}\n") 
                    all_summary[base] = results.get(base, {})
                self.root.after(0, self._finish_split, mainjob, all_summary, out_dir)
            except Exception as e:
                self.root.after(0, messagebox.showerror, "Error", f"Splitimise ajal viga: {e}")
                try:
                    self.root.after(0, self.progress_win.destroy)
                except Exception:
                    pass

        threading.Thread(target=threaded_split, daemon=True).start()
                        

    def _finish_split(self, mainjob, results, out_dir):
            try:
                self.progress_win.destroy()
            except Exception:
                pass
            parts = []
            for pdfname, data in results.items():
                for sj, files in data.items():
                    parts.append(f"{pdfname} - {sj}: {len(files)} faili")

            # <-- Algsete failide liigutamine processed kausta
            if self.processed_dir and self.pdf_list:
                for pdf in self.pdf_list:
                    try:
                        shutil.move(pdf, os.path.join(self.processed_dir, os.path.basename(pdf)))
                    except Exception as e:
                        messagebox.showwarning("Hoiatus", f"Faili {pdf} ei õnnestunud processed kausta liigutada: {e}")

            if parts:
                msg = "\n".join(parts)
                messagebox.showinfo("Done", f"Main Job '{mainjob}' splititud.\n\n{msg}\n\nLogifailid: {out_dir}")
            else:
                messagebox.showwarning("Tulemus", "Ühtegi faili ei loodud (pole ühtegi matching rule'i või lehti).")

# ---------------------------- RUN ----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = ModernPDFSplitter(root)
    root.mainloop()
