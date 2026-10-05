"""
Generate a consolidated PDF report for a training/validation run.

Usage:
    python -m speed_profile_generator.reporting.generate_pdf --run_dir runs/your_run
    
    # Or with overrides:
    python -m speed_profile_generator.reporting.generate_pdf \
        --run_dir runs/your_run \
        --output my_report.pdf \
        --report_name custom_report.md
"""

import os
import sys
import argparse
import glob
import re
import math
import yaml
from pathlib import Path
from datetime import datetime

# ReportLab imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Preformatted, Image as RLImage, Table, TableStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
except ImportError:
    print("Error: reportlab is not installed. Please install it using 'pip install reportlab' or 'conda install reportlab'")
    sys.exit(1)

def parse_markdown_line(line):
    """Simple markdown parser for bold/italics."""
    # Convert **bold** to <b>bold</b>
    line = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', line)
    # Convert *italics* to <i>italics</i>
    line = re.sub(r'\*(.*?)\*', r'<i>\1</i>', line)
    # Convert `code` to <font name="Courier">code</font>
    line = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', line)
    return line

def create_table_from_lines(lines, styles):
    """Parse markdown table lines and return a ReportLab Table."""
    data = []
    for line in lines:
        if set(line.strip()) == {'-', '|', ' '}: continue # Skip separator lines
        row = [cell.strip() for cell in line.split('|')]
        # Filter empty strings from split edges if any
        if not row[0]: row = row[1:]
        if not row[-1]: row = row[:-1]
        
        # Parse cells potentially containing bold/etc
        parsed_row = [Paragraph(parse_markdown_line(cell), styles['Normal']) for cell in row]
        data.append(parsed_row)
        
    if not data: return None
    
    # Create Table
    t = Table(data, hAlign='LEFT')
    
    # Style
    style = TableStyle([
        ('GRID', (0,0), (-1,-1), 1, colors.black),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('PADDING', (0,0), (-1,-1), 6),
    ])
    t.setStyle(style)
    return t

def generate_pdf(run_dir, output_filename="run_report_full.pdf", report_name="engineering_report.md", config_name="config.yaml"):
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Error: Run directory '{run_dir}' does not exist.")
        sys.exit(1)
        
    output_path = run_path / output_filename
    report_file = run_path / report_name
    config_file = run_path / config_name
    
    print(f"Generating PDF report for: {run_dir}")
    
    # --- Content Discovery ---
    has_config = config_file.exists()
    has_report = report_file.exists()
    
    all_plots = sorted(glob.glob(str(run_path / "validation_plots" / "**" / "*.png"), recursive=True))
    
    # Categorize Plots
    bellman_plots = [p for p in all_plots if "bellman" in os.path.basename(p).lower()]
    other_plots = [p for p in all_plots if p not in bellman_plots]
    
    print(f"Found Config: {has_config}")
    print(f"Found Report: {has_report}")
    print(f"Found Plots: {len(all_plots)} (Bellman: {len(bellman_plots)}, Other: {len(other_plots)})")
    
    # --- PDF Setup ---
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=50, leftMargin=50,
        topMargin=50, bottomMargin=50
    )
    
    story = []
    styles = getSampleStyleSheet()
    
    # Custom Styles
    if 'Code' not in styles:
        styles.add(ParagraphStyle(name='Code', parent=styles['BodyText'], fontName='Courier', fontSize=8, leading=10, spaceAfter=5))
    if 'Caption' not in styles:
        styles.add(ParagraphStyle(name='Caption', parent=styles['BodyText'], fontName='Helvetica-Oblique', fontSize=9, alignment=1, spaceAfter=10))
    if 'Warning' not in styles:
        styles.add(ParagraphStyle(name='Warning', parent=styles['BodyText'], textColor=colors.red))
    
    # --- 1. Cover Page ---
    story.append(Paragraph("Run Report", styles['Title']))
    story.append(Spacer(1, 12))
    story.append(Paragraph(f"Run Name: {run_path.name}", styles['Heading2']))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", styles['Normal']))
    story.append(Spacer(1, 24))
    
    story.append(Paragraph("Contents:", styles['Heading3']))
    story.append(Paragraph(f"- Config: {'Found' if has_config else 'Missing'}", styles['Normal']))
    story.append(Paragraph(f"- Engineering Report: {'Found' if has_report else 'Missing'}", styles['Normal']))
    story.append(Paragraph(f"- Validation Plots: {len(all_plots)}", styles['Normal']))
    story.append(PageBreak())
    
    # --- 2. Config Section ---
    story.append(Paragraph("Configuration", styles['Heading1']))
    if has_config:
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                config_content = f.read()
            story.append(Preformatted(config_content, styles['Code']))
        except Exception as e:
            story.append(Paragraph(f"Error reading config: {e}", styles['Warning']))
    else:
        story.append(Paragraph(f"{config_name} not found.", styles['Warning']))
    story.append(PageBreak())
    
    # --- 3. Engineering Report Section ---
    if has_report:
        story.append(Paragraph("Engineering Report", styles['Heading1']))
        try:
            with open(report_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                
            in_code_block = False
            code_buffer = []
            table_buffer = []
            
            for line in lines:
                line_stripped = line.strip()
                
                # Check formatting state
                is_table_row = line.strip().startswith("|") and "|" in line.strip()[1:]
                
                # Flush table buffer if not a table row
                if table_buffer and not is_table_row:
                    t = create_table_from_lines(table_buffer, styles)
                    if t: story.append(t)
                    table_buffer = []
                    
                # Code Block Handling
                if line_stripped.startswith("```"):
                    if in_code_block:
                        # End block
                        in_code_block = False
                        if code_buffer:
                            story.append(Preformatted("".join(code_buffer), styles['Code']))
                        code_buffer = []
                    else:
                        in_code_block = True
                        code_buffer = []
                    continue
                    
                if in_code_block:
                    code_buffer.append(line)
                    continue
                
                # Table Handling
                if is_table_row:
                    table_buffer.append(line)
                    continue
                
                # Markdown Parsing
                parsed_line = parse_markdown_line(line.strip())
                
                if line.startswith("# "):
                    story.append(Paragraph(parsed_line[2:].strip(), styles['Heading1']))
                elif line.startswith("## "):
                    story.append(Paragraph(parsed_line[2:].strip(), styles['Heading2']))
                elif line.startswith("### "):
                    story.append(Paragraph(parsed_line[2:].strip(), styles['Heading3']))
                elif line_stripped.startswith("!") and "](" in line_stripped:
                    # Parse Image: ![alt](path)
                    try:
                        start_idx = line_stripped.find("](") + 2
                        end_idx = line_stripped.find(")", start_idx)
                        img_rel_path = line_stripped[start_idx:end_idx]
                        
                        img_path = run_path / img_rel_path
                        if img_path.exists():
                            img = RLImage(str(img_path))
                            avail_width = A4[0] - 100
                            img_w, img_h = img.imageWidth, img.imageHeight
                            scale = min(1.0, avail_width / img_w)
                            img.drawWidth = img_w * scale
                            img.drawHeight = img_h * scale
                            story.append(img)
                        else:
                             story.append(Paragraph(f"[Image Missing: {img_rel_path}]", styles['Warning']))
                    except:
                        story.append(Paragraph(parsed_line, styles['Normal']))
                elif line_stripped == "":
                     story.append(Spacer(1, 6))
                else:
                    # Generic paragraph with formatted text
                    story.append(Paragraph(parsed_line, styles['Normal']))
            
            # Flush existing table buffer at end
            if table_buffer:
                t = create_table_from_lines(table_buffer, styles)
                if t: story.append(t)
                    
        except Exception as e:
            story.append(Paragraph(f"Error reading report: {e}", styles['Warning']))
            import traceback
            traceback.print_exc()
            
        story.append(PageBreak())

    # --- 4. Plots Appendix ---
    
    # 4a. Other Plots (Priority)
    if other_plots:
        story.append(Paragraph("General Validation Plots", styles['Heading1']))
        for p_path in other_plots:
            rel_name = os.path.relpath(p_path, run_dir)
            try:
                img = RLImage(p_path)
                avail_width = A4[0] - 100
                avail_height = A4[1] - 150 # Reserve header/caption space
                
                w, h = img.imageWidth, img.imageHeight
                scale_w = avail_width / w
                scale_h = avail_height / h
                scale = min(1.0, scale_w, scale_h)
                
                img.drawWidth = w * scale
                img.drawHeight = h * scale
                
                story.append(Paragraph(rel_name, styles['Heading3']))
                story.append(img)
                story.append(PageBreak())
            except Exception as e:
                print(f"Warning: Could not add image {rel_name}: {e}")
                
    # 4b. Bellman Plots (Grid Layout)
    if bellman_plots:
        story.append(Paragraph("Bellman Segment Analysis", styles['Heading1']))
        story.append(Paragraph(f"Total Segments: {len(bellman_plots)}", styles['Normal']))
        story.append(Spacer(1, 12))
        
        # Grid settings: 2 columns, 3 rows = 6 per page
        COLS = 2
        ROWS = 3
        PLOTS_PER_PAGE = COLS * ROWS
        
        # Calculate cell size
        page_w, page_h = A4
        margin = 50
        avail_w = page_w - 2 * margin
        avail_h = page_h - 2 * margin - 50 # Header space
        
        cell_w = avail_w / COLS
        cell_h = avail_h / ROWS
        
        def process_image_for_grid(path):
            try:
                img = RLImage(path)
                # Scale to fit cell with padding
                w, h = img.imageWidth, img.imageHeight
                scale = min((cell_w - 10)/w, (cell_h - 20)/h)
                img.drawWidth = w * scale
                img.drawHeight = h * scale
                
                # Caption
                caption = os.path.basename(path).replace("bellman_", "").replace(".png", "")
                p = Paragraph(caption, styles['Caption'])
                return [img, p]
            except:
                return [Paragraph("Error", styles['Warning'])]

        # Chunky processing
        for i in range(0, len(bellman_plots), PLOTS_PER_PAGE):
            batch = bellman_plots[i : i + PLOTS_PER_PAGE]
            
            # Prepare data for table
            # We want a table where each cell contains [Image, Caption]
            # Table structure: [ [Cell1, Cell2], [Cell3, Cell4], ... ]
            
            table_data = []
            current_row = []
            
            for p_path in batch:
                cell_content = process_image_for_grid(p_path)
                current_row.append(cell_content)
                
                if len(current_row) == COLS:
                    table_data.append(current_row)
                    current_row = []
            
            # Fill last row if incomplete
            if current_row:
                while len(current_row) < COLS:
                    current_row.append([])
                table_data.append(current_row)
                
            # Create Table
            t = Table(table_data, colWidths=[cell_w]*COLS, rowHeights=[cell_h]*len(table_data))
            t.setStyle(TableStyle([
                ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                ('ALIGN', (0,0), (-1,-1), 'CENTER'),
                #('GRID', (0,0), (-1,-1), 0.5, colors.grey), # Optional grid lines
            ]))
            story.append(t)
            story.append(PageBreak())

    if not all_plots:
        story.append(Paragraph("No validation plots found.", styles['Normal']))

    # --- Build ---
    try:
        doc.build(story)
        print("PDF generated successfully.")
    except Exception as e:
        print(f"Error building PDF: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', type=str, required=True, help="Path to the training run directory")
    parser.add_argument('--output', type=str, default="run_report_full.pdf", help="Output PDF filename (default: run_report_full.pdf inside run_dir)")
    parser.add_argument('--report_name', type=str, default="engineering_report.md", help="Name of the markdown report file")
    parser.add_argument('--config_name', type=str, default="config.yaml", help="Name of the config file")
    
    args = parser.parse_args()
    
    generate_pdf(args.run_dir, args.output, args.report_name, args.config_name)
