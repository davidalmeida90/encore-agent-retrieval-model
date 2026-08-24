from __future__ import annotations

from ingest.sec_tables import extract_sec_tables, tables_to_json, tables_to_markdown


PRODUCTS_HTML = """
<html>
  <body>
    <table>
      <tr><td></td><td></td><td></td></tr>
    </table>
    <p>Products and Services Performance</p>
    <p>The following table shows net sales by category for 2025, 2024 and 2023 (dollars in millions):</p>
    <table>
      <tr>
        <td></td><td colspan="3">2025</td><td colspan="3">Change</td>
        <td colspan="3">2024</td><td colspan="3">Change</td><td colspan="3">2023</td>
      </tr>
      <tr>
        <td colspan="3">iPhone</td><td>$</td><td>209,586</td><td></td><td>4</td><td>4</td><td>%</td>
        <td>$</td><td>201,183</td><td></td><td>&#8212;</td><td>&#8212;</td><td>%</td>
        <td>$</td><td>200,583</td><td></td>
      </tr>
      <tr>
        <td colspan="3">Services (1)</td><td></td><td>109,158</td><td></td><td>14</td><td>14</td><td>%</td>
        <td></td><td>96,169</td><td></td><td>13</td><td>13</td><td>%</td>
        <td></td><td>85,200</td><td></td>
      </tr>
    </table>
    <p>(1) Services net sales include amortization of deferred value.</p>
  </body>
</html>
"""


XBRL_HTML = """
<html>
  <body>
    <p>Consolidated Statements of Operations</p>
    <p>(In millions)</p>
    <table>
      <tr><td></td><td>2025</td><td>2024</td></tr>
      <tr>
        <td>Net sales</td>
        <td>$ <ix:nonFraction name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
          contextRef="c-12" unitRef="usd" decimals="-6" scale="6" id="f-72">416,161</ix:nonFraction></td>
        <td>$ 391,035</td>
      </tr>
    </table>
  </body>
</html>
"""


def test_extract_product_table_collapses_sec_layout_cells() -> None:
    tables = extract_sec_tables(PRODUCTS_HTML)

    assert len(tables) == 1
    table = tables[0]
    assert table.title == "Products and Services Performance"
    assert table.units == "dollars in millions"
    assert [column.label for column in table.columns] == [
        "Category",
        "2025 Sales",
        "2025 Change",
        "2024 Sales",
        "2024 Change",
        "2023 Sales",
    ]
    assert [row.label for row in table.rows] == ["iPhone", "Services (1)"]
    assert table.rows[0].cells[0].text == "$209,586"
    assert table.rows[0].cells[1].text == "4%"
    assert table.rows[0].cells[3].text == "-"
    assert table.footnotes == ["(1) Services net sales include amortization of deferred value."]
    assert "| iPhone | $209,586 | 4% | $201,183 | - | $200,583 |" in table.markdown
    assert "iPhone | iPhone | iPhone" not in table.markdown


def test_extract_xbrl_table_preserves_inline_fact_metadata() -> None:
    table = extract_sec_tables(XBRL_HTML)[0]

    assert table.title == "Consolidated Statements of Operations"
    assert table.units == "In millions"
    assert [column.label for column in table.columns] == [
        "Metric",
        "2025",
        "2024",
    ]
    revenue_cell = table.rows[0].cells[0]
    assert revenue_cell.text == "$416,161"
    assert len(revenue_cell.facts) == 1
    assert revenue_cell.facts[0].name == "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    assert revenue_cell.facts[0].context_ref == "c-12"
    assert revenue_cell.facts[0].unit_ref == "usd"


def test_extract_simple_statement_table_attaches_split_currency_cells() -> None:
    html = """
    <html>
      <body>
        <div>
          <span>Consolidated Statements of Operations</span>
          <span>(In millions)</span>
          <table>
            <tr><td></td><td colspan="3">2025</td><td colspan="3">2024</td></tr>
            <tr><td colspan="3">Products</td><td>$</td><td>307,003</td><td></td><td>$</td><td>294,866</td><td></td></tr>
          </table>
        </div>
      </body>
    </html>
    """

    table = extract_sec_tables(html)[0]

    assert table.title == "Consolidated Statements of Operations"
    assert table.units == "In millions"
    assert [column.label for column in table.columns] == ["Metric", "2025", "2024"]
    assert table.rows[0].label == "Products"
    assert [cell.text for cell in table.rows[0].cells] == ["$307,003", "$294,866"]


def test_extract_table_context_from_wrapper_siblings() -> None:
    html = """
    <html>
      <body>
        <div>Products and Services Performance</div>
        <div>The following table shows net sales by category for 2025 and 2024 (dollars in millions):</div>
        <div>
          <table>
            <tr><td></td><td colspan="3">2025</td><td colspan="3">Change</td><td colspan="3">2024</td></tr>
            <tr><td colspan="3">Mac</td><td>$</td><td>33,708</td><td></td><td>12</td><td>12</td><td>%</td><td>$</td><td>29,984</td><td></td></tr>
          </table>
        </div>
      </body>
    </html>
    """

    table = extract_sec_tables(html)[0]

    assert table.title == "Products and Services Performance"
    assert table.units == "dollars in millions"


def test_normalized_outputs_include_markdown_and_structured_json() -> None:
    tables = extract_sec_tables(PRODUCTS_HTML)

    markdown = tables_to_markdown(tables)
    payload = tables_to_json(tables)

    assert "## Products and Services Performance" in markdown
    assert "_Units: dollars in millions_" in markdown
    assert "| Services (1) | 109,158 | 14% | 96,169 | 13% | 85,200 |" in markdown
    assert payload[0]["title"] == "Products and Services Performance"
    assert payload[0]["rows"][0]["label"] == "iPhone"
    assert payload[0]["columns"][1]["label"] == "2025 Sales"


def test_extract_simple_table_uses_header_label_column() -> None:
    html = """
    <html><body>
      <p>Segment Revenue</p>
      <table>
        <tr><th>Segment</th><th>Revenue</th></tr>
        <tr><td>Products</td><td>100</td></tr>
        <tr><td>Services</td><td>50</td></tr>
      </table>
    </body></html>
    """

    table = extract_sec_tables(html)[0]

    assert [column.label for column in table.columns] == ["Segment", "Revenue"]
    assert [row.label for row in table.rows] == ["Products", "Services"]
    assert [cell.text for cell in table.rows[0].cells] == ["100"]


def test_extract_sales_change_table_accepts_compact_percent_cells() -> None:
    html = """
    <html><body>
      <p>Products and Services Performance</p>
      <table>
        <tr><td></td><td>2025</td><td>Change</td><td>2024</td></tr>
        <tr><td>Mac</td><td>33,708</td><td>12%</td><td>29,984</td></tr>
      </table>
    </body></html>
    """

    table = extract_sec_tables(html)[0]

    assert table.rows[0].label == "Mac"
    assert [cell.text for cell in table.rows[0].cells] == ["33,708", "12%", "29,984"]
